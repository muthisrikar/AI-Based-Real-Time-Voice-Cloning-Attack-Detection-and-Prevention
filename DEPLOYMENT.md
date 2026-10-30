# GCP Compute Engine Shielded VM Deployment Guide

This guide outlines the production deployment of **AegisVoice** on Google Cloud Platform (GCP) Compute Engine utilizing **Shielded VMs** with **Virtual Trusted Platform Module (vTPM)**, **Secure Boot**, and **Integrity Monitoring** to protect live call streams and prevent kernel- or boot-level tampering.

---

## 1. Architecture Overview

- **Edge Layer:** Cloud Armor (WAF & DDoS Mitigation) + External HTTPS/WSS Load Balancer.
- **Compute Layer:** Compute Engine Shielded VM (`c2-standard-4` compute-optimized instance).
- **Runtime:** Container-Optimized OS (COS) or Debian 12 with Docker running `aegisuser` (UID 10001).
- **Network SLA:** Direct VPC ingress with WebSocket persistent streaming; sub-150ms roundtrip latency budget.

---

## 2. Container Build & Push to Google Artifact Registry

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Create Artifact Registry Docker repository
gcloud artifacts repositories create aegis-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="AegisVoice Container Repository"

# 3. Configure Docker credential helper
gcloud auth configure-docker us-central1-docker.pkg.dev

# 4. Build and tag the production container
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-voice:latest .

# 5. Push image to Artifact Registry
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-voice:latest
```

---

## 3. Provisioning GCP Shielded VM via gcloud CLI

Deploy a Compute-Optimized Shielded VM instance (`c2-standard-4`) configured for high clock frequency and consistent DSP inference latency:

```bash
gcloud compute instances create aegis-voice-node-01 \
    --project=YOUR_PROJECT_ID \
    --zone=us-central1-a \
    --machine-type=c2-standard-4 \
    --network-interface=network=default,subnet=default \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --boot-disk-device-name=aegis-boot-disk \
    --tags=aegis-voice,https-server \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --metadata=gce-container-declaration="spec:
  containers:
    - name: aegis-voice-app
      image: 'us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-voice:latest'
      ports:
        - containerPort: 8000
          hostPort: 8000
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
      restartPolicy: Always"
```

### Key Shielded VM Flags Explained:
1. `--shielded-secure-boot`: Ensures system boots only authenticated software with digital signatures from trusted authorities, preventing rootkits and boot-level implants.
2. `--shielded-vtpm`: Implements a virtual TPM 2.0 chip conforming to Trusted Computing Group specifications for measured boot and cryptographic attestation.
3. `--shielded-integrity-monitoring`: Constantly logs and compares boot integrity baselines in Cloud Logging and Security Command Center.

---

## 4. Firewall Rule Configuration

Open port 8000 (or standard 443 with TLS termination):

```bash
gcloud compute firewall-rules create allow-aegis-voice-traffic \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8000,tcp:443 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=aegis-voice
```

---

## 5. Cloud Load Balancer & WebSocket Optimization

For sub-150ms live call intercept in carrier or VoIP environments:
1. **Enable HTTP/2 and WebSocket Upgrade Support** on the Backend Service:
   ```bash
   gcloud compute backend-services update aegis-backend-service \
       --global \
       --timeout=86400s \
       --protocol=HTTP
   ```
   *(Note: Set the backend service timeout to `86400s` to prevent premature WebSocket session termination on long active calls).*
2. **Cloud Armor Security Policy**: Apply rate limiting and IP reputation filtering to block malicious flood attacks targeting the audio analysis socket.

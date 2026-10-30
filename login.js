/**
 * login.js - Client-Side Authentication Controller for AegisVoice
 * Handles Sign In and Sign Up transitions, input validation, and API authentication.
 */

const tabLoginBtn = document.getElementById("tabLoginBtn");
const tabSignupBtn = document.getElementById("tabSignupBtn");
const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");
const authAlert = document.getElementById("authAlert");

const loginSubmitBtn = document.getElementById("loginSubmitBtn");
const signupSubmitBtn = document.getElementById("signupSubmitBtn");

// 1. Tab Switching
function switchAuthTab(tab) {
  clearAlert();
  if (tab === "login") {
    tabLoginBtn.classList.add("active");
    tabSignupBtn.classList.remove("active");
    loginForm.style.display = "block";
    signupForm.style.display = "none";
  } else {
    tabSignupBtn.classList.add("active");
    tabLoginBtn.classList.remove("active");
    signupForm.style.display = "block";
    loginForm.style.display = "none";
  }
}

// 2. Password Visibility Toggle
function togglePasswordVisibility(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;

  if (input.type === "password") {
    input.type = "text";
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
        <line x1="1" y1="1" x2="23" y2="23"/>
      </svg>
    `;
  } else {
    input.type = "password";
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
        <circle cx="12" cy="12" r="3"/>
      </svg>
    `;
  }
}

// 3. Alerts
function showAlert(message, type = "error") {
  authAlert.textContent = message;
  authAlert.className = `alert-banner ${type}`;
}

function clearAlert() {
  authAlert.textContent = "";
  authAlert.className = "alert-banner";
}

// 4. Handle Login Form Submit
async function handleLoginSubmit(event) {
  event.preventDefault();
  clearAlert();

  const usernameOrEmail = document.getElementById("loginUser").value.trim();
  const password = document.getElementById("loginPass").value;

  if (!usernameOrEmail || !password) {
    showAlert("Please enter both username/email and password.");
    return;
  }

  setLoading(loginSubmitBtn, true, "AUTHENTICATING...");

  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username_or_email: usernameOrEmail,
        password: password,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Authentication failed. Check your credentials.");
    }

    // Save session info
    if (data.token) {
      localStorage.setItem("aegis_token", data.token);
    }
    if (data.user) {
      localStorage.setItem("aegis_user", JSON.stringify(data.user));
    }

    showAlert("Identity verified! Entering secure console...", "success");

    // Redirect to main detection app
    setTimeout(() => {
      window.location.href = "/";
    }, 600);
  } catch (err) {
    showAlert(err.message, "error");
    setLoading(loginSubmitBtn, false, "ENTER DEFENSE CONSOLE");
  }
}

// 5. Handle Signup Form Submit
async function handleSignupSubmit(event) {
  event.preventDefault();
  clearAlert();

  const username = document.getElementById("signupUsername").value.trim();
  const email = document.getElementById("signupEmail").value.trim();
  const password = document.getElementById("signupPass").value;

  if (username.length < 3) {
    showAlert("Username must be at least 3 characters.");
    return;
  }
  if (password.length < 6) {
    showAlert("Password must be at least 6 characters.");
    return;
  }

  setLoading(signupSubmitBtn, true, "REGISTERING VAULT...");

  try {
    const response = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: username,
        email: email,
        password: password,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Account creation failed.");
    }

    if (data.token) {
      localStorage.setItem("aegis_token", data.token);
    }
    if (data.user) {
      localStorage.setItem("aegis_user", JSON.stringify(data.user));
    }

    showAlert("Account created successfully! Entering console...", "success");

    setTimeout(() => {
      window.location.href = "/";
    }, 700);
  } catch (err) {
    showAlert(err.message, "error");
    setLoading(signupSubmitBtn, false, "CREATE SECURE ACCOUNT");
  }
}

function setLoading(btn, isLoading, labelText) {
  if (isLoading) {
    btn.disabled = true;
    btn.innerHTML = `
      <svg class="spin-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spin 0.8s linear infinite;">
        <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-linecap="round" />
      </svg>
      <span>${labelText}</span>
    `;
  } else {
    btn.disabled = false;
    btn.innerHTML = `
      <span>${labelText}</span>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="5" y1="12" x2="19" y2="12"/>
        <polyline points="12 5 19 12 12 19"/>
      </svg>
    `;
  }
}

// Check if user is already logged in
async function checkExistingAuth() {
  const token = localStorage.getItem("aegis_token");
  try {
    const res = await fetch("/api/auth/me", {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (res.ok) {
      // Already authenticated, forward to dashboard
      window.location.href = "/";
    }
  } catch (e) {
    // Not authenticated, stay on login page
  }
}

document.addEventListener("DOMContentLoaded", checkExistingAuth);

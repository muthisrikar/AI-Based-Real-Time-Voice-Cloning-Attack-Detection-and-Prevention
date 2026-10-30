"""
database.py - SQLite Database Management for AegisVoice Authentication
Stores user credentials (salted PBKDF2-SHA256 password hashes) and active sessions.
Zero external dependencies; utilizes Python's built-in sqlite3 and hashlib.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with row factory configured."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return dk.hex(), salt.hex()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    """Verifies a password against the stored salt and hash."""
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return secrets.compare_digest(dk.hex(), expected_hash_hex)


def create_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Registers a new user in the database."""
    username = username.strip()
    email = email.strip().lower()
    
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    if "@" not in email or "." not in email:
        raise ValueError("Please provide a valid email address.")
        
    pwd_hash, salt = hash_password(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username, email, pwd_hash, salt)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return {"id": user_id, "username": username, "email": email}
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            raise ValueError("Username is already registered. Please choose another.")
        elif "email" in err_msg:
            raise ValueError("Email is already registered. Please sign in.")
        raise ValueError("User with these details already exists.")
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates user credentials against stored records."""
    identifier = username_or_email.strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
        (identifier, identifier)
    )
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
        
    if verify_password(password, user["salt"], user["password_hash"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        }
    return None


def create_session(user_id: int, days_valid: int = 7) -> str:
    """Generates and stores a secure session token."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=days_valid)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at.isoformat())
    )
    conn.commit()
    conn.close()
    return token


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """Validates a session token and returns associated user details."""
    if not token:
        return None
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.username, u.email, s.expires_at 
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
    """, (token,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    # Check expiration
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.utcnow() > expires_at:
        delete_session(token)
        return None
        
    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"]
    }


def delete_session(token: str):
    """Deletes a session token on logout."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# Auto-initialize database schema on module import
init_db()

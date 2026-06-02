import os
import sqlite3
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = ['temp', 'outputs', 'uploads', 'models', 'modules']

for folder in DIRS:
    path = os.path.join(BASE_DIR, folder)
    if not os.path.exists(path):
        os.makedirs(path)

# --- AWS CONFIGURATION ---
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# --- DATABASE CONFIGURATION ---
DB_PATH = os.path.join(BASE_DIR, "nexus.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # User logins table
    cursor.execute('CREATE TABLE IF NOT EXISTS users (user TEXT PRIMARY KEY, pw TEXT)')
    # Translation history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            module TEXT,
            source_lang TEXT,
            target_lang TEXT,
            original_text TEXT,
            translated_text TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

# --- AWS BOTO3 CLIENT HELPER ---
def get_aws_client(service_name):
    """
    Returns an AWS client for the requested service (e.g., polly, translate).
    Returns None if AWS credentials are not set up or invalid.
    """
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        return None
    try:
        return boto3.client(
            service_name,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
    except Exception:
        return None

# --- SUPPORTED INDIAN LANGUAGES ---
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'ur': 'Urdu'
}

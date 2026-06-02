import os
import sqlite3
import torch
import pandas as pd
from datetime import datetime
from config import DB_PATH, BASE_DIR

def save_translation_history(module, source_lang, target_lang, original_text, translated_text, confidence=1.0):
    """
    Saves an entry into the SQLite history database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (module, source_lang, target_lang, original_text, translated_text, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (module, source_lang, target_lang, original_text, translated_text, confidence))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False

def get_translation_history(limit=50):
    """
    Fetches the latest translation history logs.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f'''
            SELECT timestamp, module, source_lang, target_lang, original_text, translated_text, confidence
            FROM history
            ORDER BY timestamp DESC
            LIMIT {limit}
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error reading history: {e}")
        return pd.DataFrame()

def clear_translation_history():
    """
    Deletes all history logs.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False

def clean_directories():
    """
    Cleans up old temporary files in temp/, outputs/, and uploads/.
    """
    dirs_to_clean = ['temp', 'outputs', 'uploads']
    cleaned_count = 0
    for folder in dirs_to_clean:
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    return cleaned_count

def check_gpu_support():
    """
    Checks if a CUDA GPU is available for PyTorch.
    """
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Only"
    return gpu_available, device_name

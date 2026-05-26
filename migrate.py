import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns = [
        ("full_name", "TEXT"),
        ("gender", "TEXT"),
        ("birth_date", "TEXT"),
        ("blood_group", "TEXT"),
        ("phone", "TEXT"),
        ("address", "TEXT"),
        ("profile_completed", "INTEGER DEFAULT 0")
    ]
    
    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError:
            print(f"Column already exists: {col_name}")
            
    conn.commit()
    conn.close()
    print("Migration finished.")

if __name__ == "__main__":
    migrate()

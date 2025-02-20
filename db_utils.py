import sqlite3
from typing import List, Dict

DB_FILE = "chat_history.db"  # The .db file storing conversation

def init_db():
    """Create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_message(session_id: str, role: str, content: str):
    """Insert a single message into the conversation_messages table."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO conversation_messages (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    conn.commit()
    conn.close()

def get_conversation(session_id: str) -> List[Dict[str, str]]:
    """Retrieve all messages for a given session_id, ordered by creation time."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM conversation_messages
        WHERE session_id=?
        ORDER BY created_at ASC
    """, (session_id,))
    rows = c.fetchall()
    conn.close()
    # Convert each row into a dict {role, content} for GPT
    messages = [{"role": row[0], "content": row[1]} for row in rows]
    return messages

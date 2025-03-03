import sqlite3
from typing import List, Dict

# Database file containing conversation history.
DB_FILE = "chat_history.db"


def init_db() -> None:
    """Initialize the database by creating the conversation_messages table if it does not exist.

    The table stores:
      - session_id: Identifier for the chat session.
      - role: The role of the sender (e.g., 'user', 'assistant', or 'system').
      - content: The text content of the message.
      - created_at: Timestamp when the message was created.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def insert_message(session_id: str, role: str, content: str) -> None:
    """Insert a message into the conversation_messages table.

    Args:
        session_id (str): The unique identifier for the chat session.
        role (str): The role of the sender (e.g., 'user', 'assistant', 'system').
        content (str): The text content of the message.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO conversation_messages (session_id, role, content)
        VALUES (?, ?, ?)
        """,
        (session_id, role, content),
    )
    conn.commit()
    conn.close()


def get_conversation(session_id: str) -> List[Dict[str, str]]:
    """Retrieve the conversation history for a given session, ordered by creation time.

    Args:
        session_id (str): The unique identifier for the chat session.

    Returns:
        List[Dict[str, str]]: A list of messages as dictionaries, each containing:
            - role: The role of the sender.
            - content: The text content of the message.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        SELECT role, content FROM conversation_messages
        WHERE session_id = ?
        ORDER BY created_at ASC
        """,
        (session_id,),
    )
    rows = c.fetchall()
    conn.close()

    # Convert each row into a dictionary with keys 'role' and 'content'.
    messages = [{"role": row[0], "content": row[1]} for row in rows]
    return messages





def get_all_sessions() -> list:
    """Retrieve all chat sessions with their first user message and creation time.

    Returns:
        list: A list of dictionaries, each containing:
            - id (str): The session ID.
            - first_message (str): The first message from the user in that session.
            - created_at (str): The timestamp of the first message.
    """
    query = """
    SELECT 
        DISTINCT session_id AS id,
        MIN(CASE WHEN role = 'user' THEN content ELSE NULL END) AS first_message,
        MIN(created_at) AS created_at
    FROM 
        conversation_messages 
    GROUP BY 
        session_id
    ORDER BY 
        created_at DESC
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(query)

    # Convert query results to a list of dictionaries.
    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()
    return results


def get_conversation_with_timestamp(session_id: str) -> list:
    """Retrieve the full conversation for a given session, including role, content, and timestamp.

    Args:
        session_id (str): The unique identifier for the chat session.

    Returns:
        list: A list of dictionaries, each containing:
            - role (str): The role of the message sender.
            - content (str): The text content of the message.
            - created_at (str): The timestamp of when the message was created.
    """
    query = """
    SELECT 
        role, 
        content,
        created_at
    FROM 
        conversation_messages 
    WHERE 
        session_id = ?
    ORDER BY 
        created_at ASC
    """

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(query, (session_id,))

    # Convert query results to a list of dictionaries.
    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()
    return results

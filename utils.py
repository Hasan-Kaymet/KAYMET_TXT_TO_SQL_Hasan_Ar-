import os
import sqlite3
from typing import Dict, List
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def generate_sql_query(natural_query: str) -> str:
    """
    Convert a natural language query into an optimized SQL statement for our SQLite DB.
    Uses GPT-4 with a schema-aware prompt (Products, Transactions, Stores) and internal self-critique.
    """
    # Create a detailed system prompt with database schema and instructions for self-critique.
    system_prompt = (
        "You are an expert SQL query generator specialized in SQLite. "
        "You are provided with the following database schema:\n\n"
        "1. Products:\n"
        "   - ProductID\n"
        "   - Name (Name of product)\n"
        "   - Category1 (Men, Women, Kids)\n"
        "   - Category2 (Sandals, Casual Shoes, Boots, Sports Shoes)\n\n"
        "2. Transactions:\n"
        "   - StoreID\n"
        "   - ProductID\n"
        "   - Quantity\n"
        "   - PricePerQuantity\n"
        "   - Timestamp (y-m-d hour:minute:second)\n\n"
        "3. Stores:\n"
        "   - StoreID\n"
        "   - State (two-letter code e.g. NY, IL, TX)\n"
        "   - ZipCode\n\n"
        "When given a natural language query, generate an optimized, syntactically correct SQL query that "
        "adheres to the above schema. Internally, perform a self-critique to verify that the SQL is logically sound and "
        "free of syntax errors, but only output the final SQL query without any additional text or explanations."
    )

    user_prompt = f"Convert this natural language query into SQL: {natural_query}"
   
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],

    )
    sql_query_response = response.choices[0].message.content.strip()
    
    return sql_query_response

def execute_sql(sql: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        connection = sqlite3.connect("data.db")
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            results.append(dict(row))
        connection.close()
    except sqlite3.Error as e:
        results.append({"error": str(e)})
    return results

import os
import sqlite3
from typing import Dict, List
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def generate_sql_query(natural_query: str) -> str:#Burdaki kısım fonksiyonun ne tip return edeceğini gösteriyor.
    prompt = (
        "You are a professional, expert Sql developer. Convert the following natural language query into a "
        "Sql statement.Ensure that the Sql is good, valid and optimized.\n\n"
        f"Natural Language Query: {natural_query}\n\nSQL:"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "You are an professional, expert sql developer."},
                {"role": "user", "content": prompt},
            ],
        )
        sql_query_response = response.choices[0].message["content"].strip()
        return sql_query_response
    
    except openai.APIConnectionError as e:
        print(f"-- Error Failed to connect to OpenAI API: {e}")
        return f"-- Error generating sql statement : {str(e)}"
    
    except openai.RateLimitError as e:
        print(f"-- Error OpenAI API request exceeded rate limit: {e}")
        return f"-- Error generating sql statement : {str(e)}"
    
    except openai.APIError as e:
        print(f"-- Error OpenAI API returned an API Error: {e}")
        return f"-- Error generating sql statement : {str(e)}"

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

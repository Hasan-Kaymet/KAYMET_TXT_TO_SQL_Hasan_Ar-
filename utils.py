import os
import sqlite3
from typing import Any, Dict, List
from dotenv import load_dotenv
import json
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


def generate_sql_query(natural_query: str) -> str:
    """
    Converts a natural language query into an optimized SQL statement for our SQLite DB.
    Uses GPT-4o with a schema-aware prompt (Products, Transactions, Stores) and internal self-critique.
    """
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
    "When given a natural language query, generate an optimized, syntactically correct SQL query "
    "that adheres exactly to the above schema. Perform internal self-critique to ensure the SQL is "
    "logically sound and free of syntax errors. Output only the raw SQL statement with no additional text, "
    "and do not include any markdown formatting, code fences, or triple backticks."
)



    user_prompt = f"Convert this natural language query into SQL: {natural_query}"
   
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],

    )
    sql_query_response = response.choices[0].message.content.strip()
    
    return sql_query_response



def generate_sql_query_in_json(natural_query: str) -> Dict[str, str]:
    """
    Convert a natural language query into a valid SQL query.
    Return the output in JSON format with a single key: 'sql'.

     Args:
        natural_query (str): The natural language query to convert.

    Returns:
        Dict[str, str]: A dictionary containing a single key 'sql' with the generated SQL query.        
    
    """
    system_prompt = (
        "You are a database reporting expert specialized in SQLite. "
        "Given a natural language query, generate a valid SQL query that retrieves the required data "
        "from the database. Return your output strictly in JSON format with the key 'sql' containing only the SQL query. "
        "Do not include any extra text, markdown formatting, or code fences.\n\n"
        "Schema:\n"
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
        "Format: {\"sql\": \"<your SQL query>\"}"
    )

    user_prompt = f"Query: {natural_query}"

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    output_text = response.choices[0].message.content.strip()
    result = json.loads(output_text)
    return result


def generate_final_report(sql_query: str, aggregated_results: Dict[str, Any]) -> str:
    """
    A function that generates final report based on the sql query results.Model analyses the results and make explanations accordingly. 

    Args:
    query_request (QueryRequest): A request model containing the natural language query.

    Returns:
    Dict[str, Any]: A dictionary containing:
        - "sql": The generated SQL query as a string.
        - "results": A list of dictionaries representing the SQL query results.
        - "final_report": A detailed analysis report generated from the SQL query and its results.

    """
    aggregated_json = json.dumps(aggregated_results, ensure_ascii=False, indent=2)
    user_prompt = (
        f"SQL Query: {sql_query}\n"
        f"Aggregated SQL Results: {aggregated_json}\n\n"
        "Based solely on the SQL results above, provide a concise final report summarizing the key data insights. "
        "Do not include any additional commentary or extraneous details."
    )

    system_prompt = (
        "You are a database reporting expert. Provide a concise final report that only summarizes the data returned by the SQL query. "
        "Do not include any extra information beyond what is directly derived from the SQL output."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    final_report = response.choices[0].message.content.strip()
    return final_report


def execute_sql(sql: str) -> List[Dict[str, Any]]:
    """
    A utilty function that executes sql.

    Args:
    takes a string value(sql query)

    Returns:
    List of dictionaries (str,str)

    """
    results: List[Dict[str, Any]] = []
    connection = sqlite3.connect("data.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        results.append(dict(row))
    connection.close()
    return results

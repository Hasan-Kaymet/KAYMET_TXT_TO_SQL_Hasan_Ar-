import os
import sqlite3
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def generate_sql_query(natural_query: str) -> str:
    """Convert a natural language query into an optimized SQL statement for SQLite.

    Uses GPT-4o with a schema-aware prompt including the tables: Products, Transactions, and Stores.
    The function performs internal self-critique to ensure the generated SQL is both logically sound
    and syntactically correct.

    Args:
        natural_query (str): The natural language query provided by the user.

    Returns:
        str: The generated SQL query as a raw string with no additional formatting.
    """
    generate_sql_system_prompt = (
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
            {"role": "system", "content": generate_sql_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        top_p=1.0,
    )
    sql_query_response = response.choices[0].message.content.strip()
    return sql_query_response


def execute_sql(sql: str) -> List[Dict[str, Any]]:
    """Execute the given SQL query and return the results.

    This utility function executes a provided SQL query against the SQLite
    database stored in 'data.db' and returns the results as a list of dictionaries.
    Each dictionary corresponds to a row with column names as keys.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents
        a row from the query result.

    Raises:
        sqlite3.DatabaseError: If an error occurs during SQL execution.
    """
    results: List[Dict[str, Any]] = []
    connection = None

    try:
        connection = sqlite3.connect("data.db")
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            results.append(dict(row))
    except sqlite3.Error as e:
        raise
    finally:
        if connection:
            connection.close()

    return results



def get_explanation_and_sql(user_text: str) -> Dict[str, str]:
    """Get an explanation and an optional SQL query based on the user's request.

    This function calls GPT-4o with a system prompt that instructs the model to
    generate both a brief explanation (for conversational purposes) and an SQL query
    if the user's request requires one. The result is returned as a JSON object
    with two keys: 'explanation' and 'sql'. If no SQL query is needed, the 'sql'
    value is set to an empty string.

    Args:
        user_text (str): The user's input text.

    Returns:
        Dict[str, str]: A dictionary with keys:
            - "explanation": A short explanation addressing the user's request.
            - "sql": A valid SQL query if required; otherwise, an empty string.
    """
    get_explanation_system_prompt = (
        "You are a database-savvy assistant. The user may ask a mix of normal conversation "
        "and data queries. Provide a short explanation for any conversational portion, "
        "and if a database query is required, include it under 'sql'. "
        "Always return your result in JSON with 'explanation' and 'sql' keys. "
        "If no query is needed, set 'sql' to ''."
    )

    # Example schema detail, can be expanded
    schema_info = (
        "Schema:\n"
        "1. Products: (ProductID, Name, Category1, Category2)\n"
        "2. Transactions: (StoreID, ProductID, Quantity, PricePerQuantity, Timestamp)\n"
        "3. Stores: (StoreID, State, ZipCode)\n"
    )

    user_prompt = f"User request: {user_text}\n\n{schema_info}"

    functions = [
        {
            "name": "generate_explanation_and_sql",
            "description": "Generate both an explanation and optional SQL query for the request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "sql": {"type": "string"}
                },
                "required": ["explanation", "sql"],
                "additionalProperties": False,
            },
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": get_explanation_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        functions=functions,
        function_call={"name": "generate_explanation_and_sql"},
        temperature=0.0,
        top_p=1.0,
    )

    function_args = response.choices[0].message.function_call.arguments
    return json.loads(function_args)



def generate_final_report_no_decision(sql_query: str, db_results: List[Dict[str, Any]]) -> str:
    """Generate a final plain text report based on the SQL query and its results.

    This function performs a second-pass GPT call. It provides the SQL query and its 
    corresponding database results (in JSON format) as input and returns a concise final 
    analysis or summary in plain text. No additional JSON formatting is provided in the output.
    This function is working under the hood of no decision so the head function that call this function does not decide the user's message whether it is sql or general conversation.

    Args:
        sql_query (str): The SQL query that was executed.
        db_results (List[Dict[str, Any]]): The database results as a list of dictionaries.

    Returns:
        str: A concise plain text summary or report of the SQL query results.
    """
    final_report_system_prompt = (
        "You are a data analysis expert. Given the SQL query and its results, "
        "produce a final summary. Return just the text of the analysis, no extra JSON."
    )
    db_results_json = json.dumps(db_results, ensure_ascii=False, indent=2)
    user_prompt = (
        f"SQL Query: {sql_query}\n"
        f"DB Results: {db_results_json}\n\n"
        "Provide a concise final analysis or report about these results."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": final_report_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()



def build_sql_generation_prompt() -> str:
    """Build a system prompt for GPT to generate a valid SQL query with self-critique.

    This prompt instructs GPT to convert a natural language query into a valid, optimized SQL
    statement that adheres to the provided schema. GPT should perform an internal self-critique
    to ensure that the SQL is syntactically correct and logically consistent, then return the result
    strictly in JSON format with a single key 'sql' containing only the SQL query.

    Returns:
        str: The complete system prompt as a string.
    """
    return (
        "You are a database reporting expert specialized in SQLite. "
        "When given a natural language query, you will convert it into a valid and optimized SQL statement. "
        "Perform internal self-critique to ensure your SQL is syntactically correct, logically consistent, "
        "and adheres to the described schema. "
        "Return your output strictly in JSON format with a single key 'sql' that contains only the SQL query. "
        "Do not include any extra text or markdown formatting.\n\n"

        "Schema:\n"
        "1. Products:\n"
        "   - ProductID (INTEGER PRIMARY KEY)\n"
        "   - Name (TEXT, name of the product)\n"
        "   - Category1 (TEXT: 'Men', 'Women', or 'Kids')\n"
        "   - Category2 (TEXT: 'Sandals', 'Casual Shoes', 'Boots', 'Sports Shoes')\n\n"

        "2. Transactions:\n"
        "   - StoreID (INTEGER)\n"
        "   - ProductID (INTEGER)\n"
        "   - Quantity (INTEGER)\n"
        "   - PricePerQuantity (REAL)\n"
        "   - Timestamp (TEXT in the format 'YYYY-MM-DD HH:MM:SS')\n\n"

        "3. Stores:\n"
        "   - StoreID (INTEGER PRIMARY KEY)\n"
        "   - State (TEXT, two-letter code e.g. NY, IL, TX)\n"
        "   - ZipCode (TEXT, e.g. '10001')\n\n"

        "Perform a self-critique internally: check your SQL for correctness, referencing only "
        "the columns and tables described. If something seems off, refine your query internally, "
        "but output only the final, correct SQL.\n\n"

        "Output Format (example):\n"
        "{\"sql\": \"SELECT * FROM Products;\"}"
    )


def build_integrated_system_prompt() -> str:
    """Build the integrated system prompt for multi-functional chat and SQL generation.

    This prompt instructs the assistant to act both as a friendly conversational partner
    and as a database reporting expert specialized in SQLite. Depending on the user's request,
    it should either provide a conversational response or produce a valid SQL query (with internal
    self-critique) that adheres strictly to the provided schema. The output must be in JSON format
    with the following keys:
      - "type": Either "chat" or "sql"
      - "reply": The conversational reply or explanation
      - "query": The SQL query (if applicable; otherwise, an empty string)

    Returns:
        str: The complete system prompt as a string.
    """
    return (
        "You are both a friendly conversation assistant and a database reporting expert specialized in SQLite.\n\n"
        "When the user asks general questions, respond in a warm, human-like manner.\n"
        "When the user needs data from the DB, produce a valid SQL query referencing only the schema below.\n"
        "Perform self-critique internally to ensure correctness of your SQL and do not reveal that chain-of-thought.\n"
        "Output strictly in JSON: {\"type\":\"chat\" or \"sql\", \"reply\":\"...\", \"query\":\"...\"}.\n\n"
        "Schema:\n"
        "1. Products:\n"
        "   - ProductID (INTEGER PRIMARY KEY)\n"
        "   - Name (TEXT)\n"
        "   - Category1 (TEXT: 'Men', 'Women', 'Kids')\n"
        "   - Category2 (TEXT: 'Sandals', 'Casual Shoes', 'Boots', 'Sports Shoes')\n\n"
        "2. Transactions:\n"
        "   - StoreID (INTEGER)\n"
        "   - ProductID (INTEGER)\n"
        "   - Quantity (INTEGER)\n"
        "   - PricePerQuantity (REAL)\n"
        "   - Timestamp (TEXT 'YYYY-MM-DD HH:MM:SS')\n\n"
        "3. Stores:\n"
        "   - StoreID (INTEGER PRIMARY KEY)\n"
        "   - State (TEXT, two-letter code)\n"
        "   - ZipCode (TEXT)\n\n"
        "Remember: If the user doesn't need data, respond with type='chat' and set 'query' to ''.\n"
    )



def generate_plain_report(original_request: str, db_results: List[Dict[str, Any]]) -> str:
    """Generate a plain-language report from the SQL query results.

    This function produces a final, easy-to-understand report based on the SQL query results,
    without revealing any technical details about the SQL query or the underlying schema.
    It focuses solely on providing insights and key outcomes in plain language,
    tailored to the user's request.

    Args:
        original_request (str): The user's original natural language query.
        db_results (List[Dict[str, Any]]): The SQL query results as a list of dictionaries.

    Returns:
        str: A clear, concise report for a non-technical audience.
             If no results are provided, it returns a message indicating that no relevant data was found.
    """
    if not db_results:
        return "No relevant data was found."

    report_system_prompt = (
        "You are a helpful assistant. The user doesn't need to see any SQL or technical details. "
        "Just provide a clear, concise explanation of the data in plain language. Take the user's needs into account "
        "and tailor your report accordingly. Avoid mentioning SQL or schemas, and focus only on the final numbers or insights.\n\n"
        "For example, if the user's query is: 'How many different sandal products do we have and bring the best sellers' names?' "
        "an appropriate response might be: 'Here is an overview of the sandal sales data:\n\n"
        "- The sandal named \"Celestial\" was the top seller with a total of 10,003 units sold.\n"
        "- Following \"Celestial,\" the \"Opal\" sandal sold 9,952 units, making it the second most popular choice.\n"
        "- The \"Spirit\" sandal was also popular, with 9,704 units sold.\n"
        "- Other sandals that performed well include \"Apex\" and \"Banner,\" selling 5,296 and 5,269 units respectively.\n\n"
        "Overall, the data indicates that \"Celestial,\" \"Opal,\" and \"Spirit\" are significantly more popular compared "
        "to the rest, while other models sold between 5,000 to 5,500 units, suggesting moderate performance.'"
    )

    # Convert db_results to JSON so GPT can analyze it.
    results_json = json.dumps(db_results, ensure_ascii=False, indent=2)
    
    user_prompt = (
        f"User request: {original_request}\n\n"
        "Here is the data (already queried from the database):\n"
        f"{results_json}\n\n"
        "Summarize these results in plain language for a non-technical audience. "
        "Focus on the key outcomes and insights without mentioning SQL or schema details."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": report_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        top_p=1.0,
    )

    return response.choices[0].message.content.strip()




def merge_final_output_with_json_mode_multi_turn(partial_data: dict) -> str:
    """Merge partial data fields into a single unified 'merged_message' using GPT function calling.

    This function uses GPT to combine fields such as 'reply', 'final_report', and 'results'
    into one cohesive, user-facing text ('merged_message'). The GPT output is expected to be a JSON
    object with keys 'reply', 'final_report', 'results', and 'merged_message', from which this
    function returns the 'merged_message'.

    Args:
        partial_data (dict): A dictionary containing fields 'reply', 'final_report', and 'results'.

    Returns:
        str: The unified text contained in 'merged_message'.
    """
    # 1) Build the function schema used by GPT.
    merge_schema = build_merge_schema()

    # 2) Define the system prompt to instruct GPT on how to merge the fields.
    merge_system_prompt = (
        "You are a function that merges partial data into one cohesive 'merged_message'. "
        "Combine 'reply', 'final_report', and optionally 'results' into a single, user-facing text. "
        "Return the final text in 'merged_message'. "
        "Give the final text in the language of the 'reply'."
    )

    # 3) Prepare the data for GPT by merging partial_data with an empty 'merged_message' field.
    arguments_for_gpt = {**partial_data, "merged_message": ""}

    # 4) Build the user message including the partial data.
    user_message = {
        "role": "user",
        "content": (
            "Please merge the fields (reply, final_report, results) into 'merged_message'.\n"
            f"Here is the JSON:\n{json.dumps(arguments_for_gpt, ensure_ascii=False, indent=2)}"
        )
    }

    # 5) Call GPT in function-calling mode, specifying the desired function call.
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": merge_system_prompt},
            user_message
        ],
        functions=merge_schema,
        function_call={"name": "merge_final_output"},
        temperature=0.0,
    )

    # 6) Parse GPT's function call arguments from the assistant message.
    merged_data_str = response.choices[0].message.function_call.arguments
    merged_data = json.loads(merged_data_str)
    print(merged_data)

    # Expect merged_data to contain:
    # { "reply": "...", "final_report": "...", "results": [...], "merged_message": "..." }
    return merged_data["merged_message"]

def build_merge_schema() -> list:
    """Build the GPT function schema for merging partial output fields.

    This schema instructs GPT to combine fields such as 'reply', 'final_report',
    and 'results' into a single field called 'merged_message'. The schema requires
    that both 'reply' and 'merged_message' be provided in the output.

    Returns:
        list: A list containing a single dictionary defining the merge function schema.
    """
    return [
        {
            "name": "merge_final_output",
            "description": "Combine partial fields (reply, final_report, results) into 'merged_message'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {"type": "string"},
                    "final_report": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {}
                    },
                    "merged_message": {
                        "type": "string",
                        "description": "The final merged text."
                    }
                },
                "required": ["reply", "merged_message"],
                "additionalProperties": True
            }
        }
    ]

def is_read_only_sql(sql: str) -> bool:
    """Determine if the provided SQL query is read-only.

    This function performs a basic check to determine if the SQL query contains any
    forbidden keywords that could modify the database. The forbidden keywords include
    INSERT, UPDATE, DELETE, DROP, CREATE, and ALTER. The check is case-insensitive.

    Args:
        sql (str): The SQL query to check.

    Returns:
        bool: True if the SQL query is read-only, False otherwise.
    """
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
    upper_sql = sql.upper()
    return not any(keyword in upper_sql for keyword in forbidden)


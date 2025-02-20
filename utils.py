import os
import sqlite3
from typing import Any, Dict, List
import json
from dotenv import load_dotenv
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
    "logically sound and free of syntax errors. Output only the raw SQL statement with no additional text,"
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

#Sql runlamak için generic fonksiyon.
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

def get_explanation_and_sql(user_text: str) -> Dict[str, str]:
    """
    GPT call returning both an 'explanation' (for any general discussion) 
    and a 'sql' (if needed). 
    If the user also wants data from the DB, GPT includes a valid SQL statement.
    If no DB query is needed, 'sql' can be empty.
    """
    system_prompt = (
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
            "name": "generateExplanationAndSQL",
            "description": "Generate both an explanation and optional SQL query for the request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "sql": {"type": "string"}
                },
                "required": ["explanation", "sql"],
                "additionalProperties": False
            }
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        functions=functions,
        function_call={"name": "generateExplanationAndSQL"},
    )

    function_args = response.choices[0].message.function_call.arguments
    return json.loads(function_args)


def generate_final_report_no_decision(sql_query: str, db_results: List[Dict[str, Any]]) -> str:
    """
    2nd pass GPT call that sees the sql_query and db_results in JSON,
    returning a final summary in plain text. 
    If you prefer function-calling, you can do that instead.
    """
    system_prompt = (
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()



#Sql üretimi için prompt
def build_sql_generation_prompt() -> str:
    """
    Builds a system prompt for GPT to generate a valid SQL query with self-critique,
    given a natural language query and a detailed schema description.
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
    """
    Returns the big system prompt that handles both chat and SQL generation with self-critique.
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

def build_function_schema_assistant() -> list:
    """
    GPT function schema for returning {type, reply, query}.
    """
    return [
        {
            "name": "handleUserRequest",
            "description": "Generate a final response with or without SQL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "'chat' or 'sql'."
                    },
                    "reply": {
                        "type": "string",
                        "description": "User-facing text."
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL if needed, else empty."
                    }
                },
                "required": ["type", "reply", "query"]
            }
        }
    ]


def generate_merged_response(
    original_request: str, 
    explanation: str, 
    sql_query: str, 
    db_results: List[Dict[str, Any]]
) -> str:
    """
    Optionally merges the original request, the partial explanation, and 
    the DB results into a single final output (2nd GPT pass).
    """
    if not db_results:
        return explanation  # no results, just return the partial explanation

    system_prompt = (
        "You are a helpful assistant. Combine the user's request, the partial explanation, "
        "and the DB results into one cohesive answer. Provide it in plain text."
    )

    user_message = (
        f"User's original request: {original_request}\n"
        f"Partial explanation: {explanation}\n"
        f"SQL Query: {sql_query}\n"
        f"DB Results: {json.dumps(db_results, ensure_ascii=False, indent=2)}\n\n"
        "Please provide a final combined answer, referencing all relevant details."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content.strip()


def generate_plain_report(original_request: str, db_results: List[Dict[str, Any]]) -> str:
    """
    Produces a final, plain-language report from the SQL query results,
    without revealing any technical details about SQL or schema.
    """
    if not db_results:
        return "No relevant data was found."

    system_prompt = (
        "You are a helpful assistant. The user doesn't need to see any SQL or technical details. "
        "Just provide a clear, concise explanation of the data in plain language.Take the users needs into account and apply your report accordingt to message of user and data. "
        "Avoid mentioning SQL or schemas. Focus only on the final numbers or insights."

        """For example User's query : How many different sandal products we have and bring the best sellers names.
        Response : Here is an overview of the sandal sales data:\n\n- The sandal named \"Celestial\" was the top seller with a total of 10,003 units sold.\n- Following \"Celestial,\" the \"Opal\" sandal sold 9,952 units, making it the second most popular choice.\n- The \"Spirit\" sandal was also popular, with 9,704 units sold.\n- Other sandals that performed well include \"Apex\" and \"Banner,\" selling 5,296 and 5,269 units respectively.\n\nOverall, the data indicates that \"Celestial,\" \"Opal,\" and \"Spirit\" are significantly more popular compared to the rest. Most other sandal models sold between 5,000 to 5,500 units, suggesting moderate sales performance. If you're analyzing sandal sales, you might consider \"Celestial\" and \"Opal\" as standout performers in this category.

        """
    )

    # Convert db_results to JSON so GPT can analyze it:
    results_json = json.dumps(db_results, ensure_ascii=False, indent=2)
    
    user_prompt = (
        f"Here is the users request:{original_request}\n"
        f"{results_json}\n\n"
        f"Here is the data (already queried from the database):\n"
        f"{results_json}\n\n"
        "Summarize these results in plain language for a non-technical audience.Do your interperetaiton based on the needs of user. "
        "Do not mention SQL or schema details. Just the outcome."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()

def merge_final_output_with_json_mode(existing_json: dict) -> str:
    """
    Use GPT function calling to feed an existing JSON (reply, final_report, etc.)
    and let GPT produce a 'merged_message' result.
    """
    # 1) The function schema
    merge_schema = [
        {
            "name": "mergeFinalOutput",
            "description": "Merge the 'reply' and 'final_report' fields into one final plain-text message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {
                        "type": "string",
                        "description": "Partial conversation."
                    },
                    "final_report": {
                        "type": "string",
                        "description": "Summarized data or analysis."
                    },
                    "merged_message": {
                        "type": "string",
                        "description": "The final merged message in plain text."
                    }
                },
                "required": ["reply", "final_report","merged_message"],
                "additionalProperties": True
            }
        }
    ]

    # 2) We'll instruct GPT in the system prompt not to show the JSON, 
    #    but produce only 'merged_message'.
    system_prompt = (
        "You are a helpful assistant that merges partial conversation and final_report "
        "into one cohesive plain-text message. Do not reveal SQL or technical details. "
        "Only fill out 'merged_message' with the final text, referencing 'reply' and 'final_report'."
        "No need to explain steps of calculation the data just explain the info we get."
    )

    # 3) We'll pass the existing JSON object as if we are "calling" the function with known arguments.
    #    But the key we want GPT to fill is 'merged_message'.
    arguments_for_gpt = {**existing_json, "merged_message": ""}  # So GPT sees current data, but 'merged_message' is blank.

    # 4) We create a user message that indicates we want GPT to fill in the function call
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Please merge 'reply' and 'final_report' into 'merged_message'. "
                "Here is the JSON:\n"
                f"{json.dumps(arguments_for_gpt, indent=2)}"
            ),
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=merge_schema,
        function_call={"name": "mergeFinalOutput"},
    )

    # 5) GPT should respond with function_call arguments that include the merged_message
    merged_data_str = response.choices[0].message.function_call.arguments
    print(merged_data_str)
    merged_data = json.loads(merged_data_str)

    #print("12312")

    # 'merged_data' should have { "reply": "...", "final_report": "...", "merged_message": "..." }
    return merged_data["merged_message"]


def is_read_only_sql(sql: str) -> bool:
    # Very simple check
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
    upper_sql = sql.upper()
    return not any(keyword in upper_sql for keyword in forbidden)

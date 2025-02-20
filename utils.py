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
        temperature=0.0,
        top_p=1.0,

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
        temperature=0.0,
        top_p=1.0,
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
        temperature=0.0,
        top_p=1.0,
    )

    return response.choices[0].message.content.strip()



def merge_final_output_with_json_mode_multi_turn(partial_data: dict) -> str:
    """
    Uses GPT function calling to merge fields like 'reply', 'final_report', and 'results'
    into a single final 'merged_message' without manually stitching text in Python.

    partial_data might be something like:
    {
      "reply": "Here's the best-selling product.",
      "final_report": "Best seller is Orbit with 15,265 units sold.",
      "results": [ { "Name":"Orbit", "TotalQuantitySold":15265 } ]
    }

    We expect GPT to return {
      "reply":"...",
      "final_report":"...",
      "results": [...],
      "merged_message":"<unified text>"
    }
    and we return merged_message.
    """

    # 1) The function schema
    merge_schema = build_merge_schema()

    # 2) The system prompt instructing GPT how to merge
    system_prompt = (
        "You are a function that merges partial data into one cohesive 'merged_message'. "
        "Combine 'reply', 'final_report', and optionally 'results' into a single, user-facing text. "
        "Return the final text in 'merged_message'."
        "Give the final text in the language of reply."
    )

    # We pass partial_data plus an empty 'merged_message' so GPT can fill it
    arguments_for_gpt = {**partial_data, "merged_message": ""}

    # 3) We'll provide the partial data in the user content (NOT function_call on user)
    #    so the system sees it. But we do NOT attach function_call to a user message.
    user_message = {
        "role": "user",
        "content": (
            "Please merge the fields (reply, final_report, results) into 'merged_message'.\n"
            f"Here is the JSON:\n{json.dumps(arguments_for_gpt, ensure_ascii=False, indent=2)}"
        )
    }

    # 4) Call GPT in function-calling mode, specifying the function_call at top level
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            user_message
        ],
        functions=merge_schema,
        function_call={"name": "mergeFinalOutput"},  # We ask GPT to call mergeFinalOutput
        temperature=0.0,

    )
    
    # 5) Parse GPT's function call arguments from the assistant role
    merged_data_str = response.choices[0].message.function_call.arguments
    
    merged_data = json.loads(merged_data_str)
    print(merged_data)

    # 'merged_data' should have { "reply":"...", "final_report":"...", "results":..., "merged_message":"..." }
    return merged_data["merged_message"]


def build_merge_schema() -> list:
    return [
        {
            "name": "mergeFinalOutput",
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
    # Very simple check
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
    upper_sql = sql.upper()
    return not any(keyword in upper_sql for keyword in forbidden)

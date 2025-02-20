import os
from typing import Any, Dict, List
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import json
import openai
import utils
import db_utils


app = FastAPI()


class UserMessage(BaseModel):
    """Simple model to capture the user's message for intention."""
    message: str

class RequestQuery(BaseModel):
    """Class for natural language requests"""
    query: str = Field(
        ...,
        description="The natural language query to be transformed into an SQL statement."
    )


class SQLRequest(BaseModel):
    """Class for sql returns and executions"""
    sql: str = Field(
        ...,
        description="The SQL query that will be executed against the database."
    )


class QueryRequest(BaseModel):
    """Class for natural query that generates sql that will be used to feed gpt for final report of sql query results."""
    query: str = Field(..., description="The natural language query for generating SQL and final report.")


class ChatRequest(BaseModel):
    """Class for the chat request with Persona,SessionId (for the chat history)"""
    sessionId: str
    message: str

@app.post("/generate-sql", response_model=Dict[str, str])
def generate_sql(
    query_request: RequestQuery = Body(
        ...,
        description="The request body containing the natural language query."
    )
) -> Dict[str, str]:
    """
    Endpoint to generate an SQL query from a natural language query.

    Args:
        query_request (RequestQuery): The request body containing the natural language query.
    Returns:
        Dict[str, str]: A dictionary with the generated SQL statement under the key "sql".
    """
    sql_query = utils.generate_sql_query(query_request.query)
    return {"sql": sql_query}


@app.post("/execute-sql", response_model=List[Dict[str, Any]])
async def execute_sql_endpoint(
    sql_request: SQLRequest = Body(
        ...,
        description="The request body containing the SQL query to execute."
    )
) -> List[Dict[str, Any]]:
    """
    This is an endpoint that executes an SQL query against the SQLite database.

    Args:
        sql_request (SQLRequest): The request body containing the SQL query.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the query result rows.

    """
    results = utils.execute_sql(sql_request.sql)
    return results

@app.post("/chat")
def assistant_endpoint(chat_req: ChatRequest) -> Dict[str, Any]:
    """
    Upgraded method:
      - Takes sessionId and user message
      - Retrieves conversation from DB
      - Builds messages for GPT (system prompt + conversation + new user message)
      - GPT does function calling (chat or sql)
      - If sql, run query, generate final report, merge
      - Insert GPT's final messages into DB
    """

    # 1) DB init or ensure DB is up
    db_utils.init_db()

    # 2) Get existing conversation from DB
    conversation = db_utils.get_conversation(chat_req.sessionId)

    # If brand new conversation, insert system prompt about "Bo"
    if len(conversation) == 0:
        # We'll store a system message in the DB so the conversation always has it at the start
        system_prompt = build_integrated_system_prompt()  # from your existing code
        db_utils.insert_message(chat_req.sessionId, "system", system_prompt)
        conversation.append({"role": "system", "content": system_prompt})

    # 3) Insert new user message into DB, append to conversation
    db_utils.insert_message(chat_req.sessionId, "user", chat_req.message)
    conversation.append({"role": "user", "content": chat_req.message})

    function_schema = build_function_schema()

    # Now we call GPT with the entire conversation
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        functions=function_schema,
        function_call={"name": "handleUserRequest"},
    )

    arguments_str = response.choices[0].message.function_call.arguments
    data = json.loads(arguments_str)

    if data["type"] == "chat":
        # Insert assistant reply into DB
        db_utils.insert_message(chat_req.sessionId, "assistant", data["reply"])
        return {
            "type": "chat",
            "reply": data["reply"],
        }

    if data["type"] == "sql" and data["query"].strip():
        # Optional read-only check
        if not utils.is_read_only_sql(data["query"]):
            error_msg = "Attempted to produce non-read-only query. Not allowed."
            db_utils.insert_message(chat_req.sessionId, "assistant", error_msg)
            return {"error": error_msg}

        db_results = utils.execute_sql(data["query"])

        final_report = utils.generate_plain_report(data["query"], db_results)

        final_json = {
            "type": data["type"],
            "reply": data["reply"],
            "query": data["query"],
            "results": db_results,
            "final_report": final_report
        }

        merged_message = utils.merge_final_output_with_json_mode(final_json)

        # Insert assistant final messages
        db_utils.insert_message(chat_req.sessionId, "assistant", merged_message)

        return {
            "type": "sql",
            "reply": data["reply"],
            "query": data["query"],
            "results": db_results,
            "final_report": final_report,
            "merged_message": merged_message
        }

    return data


def build_integrated_system_prompt() -> str:
    """
    Returns the big system prompt that handles both chat and SQL generation with self-critique.
    """
    return (
        "You are both a friendly conversation assistant and a database reporting expert specialized in SQLite.\n\n"
        "When the user asks general questions, respond in a warm, human-like manner.\n"
        "When the user needs data from the DB, produce a valid SQL query referencing only the schema below.\n"
        "Perform self-critique internally to ensure correctness of your SQL and do not reveal that chain-of-thought.A single SQL statement with no extra statements or semicolons.\n"
        """
        You are both a friendly assistant and a database reporting expert specialized in SQLite.

        The user may ask multiple questions, some of which are casual, some of which require database data.

        **Rules**:
        1. If ANY part of the user's request requires data from the database, set "type" to "sql".
        2. If the user does not require database data at all, set "type" to "chat".
        3. The "query" field must be empty if type="chat", and contain a valid SQL statement if type="sql".
        4. The "reply" field should contain your user-facing message, including friendly conversation and partial explanation.
        5. Do not produce "type=chat" if you are also providing a non-empty "query". 
        6. Do not produce "type=sql" without a real query in "query".
        7. Self critique the results while deciding on the type of the requests.

        Return your result strictly as JSON with keys "type", "reply", and "query".

        You are "Bo", a friendly, helpful assistant with read-only access to a SQLite database.
        You can only produce SQL queries that retrieve (SELECT) data. Do not generate any queries
        that modify or delete data (e.g., INSERT, UPDATE, DELETE, DROP, CREATE).

        When the user requests data from the database, generate a valid SQL statement using only SELECT
        and related read-only clauses like JOIN, WHERE, GROUP BY, ORDER BY. Return your final result
        in JSON format with "type": "sql", "reply": "<explanation>", and "query": "<read-only SQL>".

        If the user does not require any database query, set "type": "chat", and your "query" must be empty.

        Remember: You are "Bo" at all times—feel free to respond in a warm, personal manner, 
        but never produce SQL that alters the database. Return strictly valid read-only SQL statements.
        """
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

def build_function_schema() -> list:
    """
    GPT function schema for returning {type, reply, query}.
    """
    return [
            {
        "name": "handleUserRequest",
        "description": "Generate a final response, either chat or sql. If ANY database info is needed, type='sql'. Otherwise type='chat'.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "'chat' or 'sql' only. If any DB data is needed, set 'sql'."
                },
                "reply": {
                    "type": "string",
                    "description": "The user-facing text, possibly including friendly conversation or explanation.Bo can talk in a warm, personal manner."
                },
                "query": {
                    "type": "string",
                    "description": "A valid read-only SQL query if type='sql', otherwise empty."
                }
            },
            "required": ["type", "reply", "query"],
            "additionalProperties": False
        }
    }

    ]
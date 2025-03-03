"""Module for the FastAPI application that handles SQL execution and chat requests.

This module defines API endpoints and request models to process SQL queries, natural language
chat requests, and to generate SQL for final reporting.
"""

import json
import os
from typing import Any, Dict, List

import openai
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db_utils
import utils


# Maximum iterations allowed for multi-turn interactions.
MULTI_TURN_ITERATION_MAX = 8

# Initialize the FastAPI application.
app = FastAPI()

# Origins allowed for CORS.
ORIGINS = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SQLRequest(BaseModel):
    """Represents a request to execute a SQL query.

    Attributes:
        sql: The SQL query to be executed against the database.
    """
    sql: str = Field(
        ...,
        description="The SQL query that will be executed against the database."
    )


class ChatRequest(BaseModel):
    """Represents a chat request with persona and session history.

    Attributes:
        sessionId: The unique identifier for the chat session.
        message: The message from the user.
    """
    sessionId: str
    message: str


class QueryRequest(BaseModel):
    """Represents a natural language query to generate SQL for final reporting.

    Attributes:
        query: The natural language query used to generate SQL.
    """
    query: str = Field(
        ...,
        description=("The natural language query for generating SQL and final report "
                     "of SQL query results.")
    )


class RequestQuery(BaseModel):
    """Represents the request body containing a natural language query.

    Attributes:
        query_request: The natural language query.
    """
    query_request: str = Field(
        ...,
        description="The request body containing the natural language query."
    )



@app.post("/generate-sql", response_model=Dict[str, str])
def generate_sql(
    query_request: RequestQuery = Body(
        ...,
        description="The request body containing the natural language query."
    )
) -> Dict[str, str]:
    """Generate an SQL query from a natural language query.

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
    """Execute an SQL query against the SQLite database.

    Args:
        sql_request (SQLRequest): The request body containing the SQL query.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the query result rows.
    """
    results = utils.execute_sql(sql_request.sql)
    return results


@app.post("/chat")
def assistant_endpoint(chat_req: ChatRequest) -> Dict[str, Any]:
    """Handle a multi-turn chat conversation with GPT integration.

    This endpoint:
      - Takes a session ID and user message.
      - Repeatedly calls GPT until it returns 'type=done' or 'type=chat' with no more queries.
      - For each 'sql' turn, runs the query, appends results, and calls GPT again.
      - Collects each query and result in `sql_history` and returns them at the end.

    Args:
        chat_req (ChatRequest): The request body containing sessionId and user message.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - type: The final response type (e.g., 'chat', 'sql', or 'done').
            - final_message: The unified final message for the conversation.
            - last_query: The last executed SQL query (if any).
            - last_results: The results of the last SQL query (if any).
            - sql_history: A list of all SQL queries and corresponding results from each turn.
            - turns_executed: The total number of turns executed.
    """
    # Initialize the database.
    db_utils.init_db()

    # 1) Retrieve conversation history from the database.
    conversation = db_utils.get_conversation(chat_req.sessionId)

    # If this is a brand new session, insert the system prompt.
    if not conversation:
        integrated_system_prompt = build_integrated_system_prompt()  # Includes logic for 'chat', 'sql', and 'done'.
        db_utils.insert_message(chat_req.sessionId, "system", integrated_system_prompt)
        conversation.append({"role": "system", "content": integrated_system_prompt})

    # 2) Insert the user's message.
    db_utils.insert_message(chat_req.sessionId, "user", chat_req.message)
    conversation.append({"role": "user", "content": chat_req.message})

    # Initialize variables for final output.
    final_type = None
    final_message = ""
    final_query = ""
    final_results = []
    turn_count = 0
    done = False

    # List to collect all SQL queries and results produced during multi-turn.
    sql_history = []

    # Process multi-turn conversation until a termination condition is met.
    while not done and turn_count < MULTI_TURN_ITERATION_MAX:
        turn_count += 1

        # Build the function schema (includes types: 'chat', 'sql', 'done').
        function_schema = build_function_schema()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation,
            functions=function_schema,
            function_call={"name": "handle_user_request"},
            temperature=0.0,
            top_p=1.0,
        )

        # Parse GPT's response.
        arguments_str = response.choices[0].message.function_call.arguments
        data = json.loads(arguments_str)

        gpt_type = data.get("type", "chat")
        gpt_reply = data.get("reply", "")
        gpt_query = data.get("query", "")

        final_type = gpt_type
        final_query = gpt_query

        # Insert GPT's raw JSON response into the conversation.
        raw_json_str = json.dumps(data, ensure_ascii=False)
        db_utils.insert_message(chat_req.sessionId, "assistant", raw_json_str)
        conversation.append({"role": "assistant", "content": raw_json_str})

        # ===== CASE A: type='chat' =====
        if gpt_type == "chat":
            final_message = gpt_reply
            done = True
            break

        # ===== CASE B: type='sql' =====
        elif gpt_type == "sql" and gpt_query.strip():
            # Enforce read-only SQL queries.
            if not utils.is_read_only_sql(gpt_query):
                final_message = "Error: Attempted non-read-only query."
                final_type = "chat"
                done = True
                break

            # Execute the SQL query.
            db_results = utils.execute_sql(gpt_query)
            final_results = db_results

            # Generate a plain-language summary of the SQL results.
            final_report = utils.generate_plain_report(gpt_query, db_results)

            # Merge GPT reply with query details and results.
            partial_json = {
                "reply": gpt_reply,
                "query": gpt_query,
                "results": db_results,
                "final_report": final_report,
            }
            merged_message = utils.merge_final_output_with_json_mode_multi_turn(partial_json)
            final_message = merged_message

            # Save GPT's merged response as an assistant message.
            db_utils.insert_message(chat_req.sessionId, "assistant", final_message)
            conversation.append({"role": "assistant", "content": final_message})

            # Store the current turn's query and results in the SQL history.
            sql_history.append({
                "turn": turn_count,
                "query": gpt_query,
                "results": db_results,
                "steps": merged_message,
            })

            # Insert the database results as a system message for context in the next turn.
            results_str = json.dumps({"query_results": db_results}, ensure_ascii=False)
            db_utils.insert_message(chat_req.sessionId, "system", results_str)
            conversation.append({"role": "system", "content": results_str})

            # Continue to the next iteration.
            continue

        # ===== CASE C: type='done' =====
        elif gpt_type == "done":
            final_message = gpt_reply
            done = True
            break

        else:
            final_message = "Got an unexpected response from GPT."
            done = True

    # End of multi-turn processing.

    return {
        "type": final_type,            # Final response type: 'chat', 'sql', or 'done'.
        "final_message": final_message,  # Consolidated final message.
        "last_query": final_query,       # Last SQL query executed (if any).
        "last_results": final_results,   # Results of the last SQL query (if any).
        "sql_history": sql_history,      # Complete history of SQL queries and results.
        "turns_executed": turn_count,    # Number of turns executed.
    }


def build_integrated_system_prompt() -> str:
    """Build and return the integrated system prompt for multi-turn chat and SQL generation.

    This prompt instructs the assistant on:
      - Handling multi-turn conversations,
      - Generating read-only SQL queries (SELECT statements only),
      - Responding in the user's language,
      - And returning a structured JSON output.

    The prompt also defines the database schema and the rules for the response types:
      - 'chat': No database query is required.
      - 'sql': A single valid read-only SQL query is needed.
      - 'done': The multi-turn conversation is complete.

    Returns:
        str: The complete system prompt as a string.
    """
    return (
    "You are 'Bo', a friendly, helpful assistant with read-only access to a SQLite database.\n\n"

    "You can only produce SQL queries that retrieve (SELECT) data—never modifying or deleting data. "
    "When the user requests data from the database, you must generate a valid single SQL query, referencing "
    "You always respond in the same language the user uses. If the user writes in Turkish, respond in Turkish. "
    "If they write in English, respond in English, etc.\n\n"  
    "only the schema below. Use SELECT, JOIN, WHERE, GROUP BY, ORDER BY, etc. for read-only. "
    "No INSERT, UPDATE, DELETE, DROP, or CREATE.\n\n"

    """

    "Database Schema:\n"
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


    """

    "Your possible 'type' values:\n"
    "  - 'chat': If the user's request does not require a database query.\n"
    "  - 'sql': If any part of the request requires data from the DB, return exactly one valid SQL query.\n"
    "  - 'done': If you have completed a multi-turn process and no more queries or conversation are needed.When you produce type='done', unify all relevant data from earlier steps (e.g. best and worst sellers). Ensure the final message references both so the user is not missing any details.”\n\n"

    "Rules:\n"
    "1. If ANY part of the user's request (or your logic) needs data from the DB, set 'type'='sql'.\n"
    "2. If the user doesn't require any DB data, set 'type'='chat' with 'query'=''.\n"
    "3. If no further queries or conversation are needed, set 'type'='done'.\n"
    "4. 'query' must be empty if 'type'='chat' or 'type'='done'.\n"
    "5. 'query' must have a single valid read-only SQL statement if 'type'='sql'.\n"
    "6. Self-critique your SQL to ensure correctness. Output no chain-of-thought.\n"
    "7. Return your final result strictly in JSON with keys 'type', 'reply', 'query'.\n\n"

    "Important:\n"
    "- If a single user request needs multiple data points, you can produce one query at a time, see the results, "
    "and possibly produce another query in a new turn. Only produce 'type'='done' after all queries are concluded.\n"
    "- ORDER BY clause should come after UNION, not before (avoid syntax errors).\n\n"

    "Remember:\n"
    " - 'type':'chat'  => 'query':''\n"
    " - 'type':'sql'   => 'query': a single read-only SQL statement\n"
    " - 'type':'done'  => 'query':'' if everything is finished.\n\n"
)



def build_function_schema() -> list:
    """Build and return the GPT function schema for handling user requests.

    The schema defines the expected JSON output with the keys:
      - 'type': indicating the response type ('chat', 'sql', or 'done'),
      - 'reply': a conversational reply or explanation,
      - 'query': a valid read-only SQL query when 'type' is 'sql' (empty otherwise).

    This schema is used in a multi-turn endpoint where:
      - 'chat' means only a conversation reply (no SQL query),
      - 'sql' means a single valid read-only SQL statement is required,
      - 'done' indicates that no further queries or conversation are needed.

    Returns:
        list: A list containing a single dictionary defining the function schema.
    """
    return [
        {
            "name": "handle_user_request",
            "description": (
                "Give the response in the language of the user."
                "Generate a final response, which can be 'chat', 'sql', or 'done'. "
                "If any database data is needed, set 'type'='sql'. If no DB data is required, "
                "set 'type'='chat'. If no further queries or conversation are needed, set 'type'='done'. "
                "The field 'query' should be empty unless 'type'='sql', in which case it should be one valid "
                "read-only SQL statement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "'chat', 'sql', or 'done'."
                    },
                    "reply": {
                        "type": "string",
                        "description": (
                            "A user-facing text or explanation. Bo can respond in a warm, personal manner. "
                            "If 'type'='sql', this might also describe the purpose of the query. "
                            "If 'type'='chat' or 'done', this is just the final conversation text."
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "A valid read-only SQL query if 'type' is 'sql'. Otherwise, an empty string."
                        )
                    }
                },
                "required": ["type", "reply", "query"],
                "additionalProperties": False
            }
        }
    ]



@app.get("/chat_sessions")
def get_chat_sessions() -> Dict[str, Any]:
    """Retrieve a list of all chat sessions along with their first message.

    Returns:
        Dict[str, Any]: A dictionary containing a list of sessions.
    """
    sessions = db_utils.get_all_sessions()
    return {"sessions": sessions}


@app.get("/chat_history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    """Retrieve the full conversation history for a given session ID.

    The returned history filters out system messages for frontend display,
    showing only user and assistant messages.

    Args:
        session_id (str): The session identifier.

    Returns:
        Dict[str, Any]: A dictionary containing the session ID and a list of messages.
    """
    conversation = db_utils.get_conversation_with_timestamp(session_id)

    # Filter out system messages for frontend display.
    user_assistant_messages = [
        msg for msg in conversation if msg["role"] in ["user", "assistant"]
    ]

    return {
        "sessionId": session_id,
        "messages": user_assistant_messages,
    }



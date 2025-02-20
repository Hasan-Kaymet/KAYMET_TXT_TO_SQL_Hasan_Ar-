import os
import json
from typing import Any, Dict, List
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import openai
import utils
import db_utils

MULTI_TURN_ITERATION_MAX = 8
app = FastAPI()



class SQLRequest(BaseModel):
    """Class for sql returns and executions"""
    sql: str = Field(
        ...,
        description="The SQL query that will be executed against the database."
    )

class ChatRequest(BaseModel):
    """Class for the chat request with Persona,SessionId (for the chat history)"""
    sessionId: str
    message: str

class QueryRequest(BaseModel):
    """Class for natural query that generates sql that will be used to feed gpt for final report of sql query results."""
    query: str = Field(..., description="The natural language query for generating SQL and final report.")


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


#Merged olarak hepsi birleşitirildi iteration sayısı max 8 
@app.post("/chat")
def assistant_endpoint(chat_req: ChatRequest) -> Dict[str, Any]:
    """
    A single endpoint that:
      - Takes sessionId and user message
      - Repeatedly calls GPT until it returns 'type=done' or 'type=chat' with no more queries
      - For each 'sql' turn, runs the query, appends results, calls GPT again
      - Collects each query & result in 'sql_history' and returns them at the end
    """

    db_utils.init_db()
    
    # 1) Retrieve conversation from DB
    conversation = db_utils.get_conversation(chat_req.sessionId)

    # If brand new session, insert system prompt
    if len(conversation) == 0:
        system_prompt = build_integrated_system_prompt()  # includes 'chat','sql','done' logic
        db_utils.insert_message(chat_req.sessionId, "system", system_prompt)
        conversation.append({"role": "system", "content": system_prompt})

    # 2) Insert user message
    db_utils.insert_message(chat_req.sessionId, "user", chat_req.message)
    conversation.append({"role": "user", "content": chat_req.message})

    # We'll store final outputs in these variables
    final_type = None
    final_message = ""
    final_query = ""
    final_results = []
    turn_count = 0
    done = False

    # A list to collect all SQL queries & results produced during multi-turn
    sql_history = []

    while not done and turn_count < MULTI_TURN_ITERATION_MAX:  # prevent infinite loop
        turn_count += 1
        
        function_schema = build_function_schema()  # includes type='chat','sql','done'
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation,
            functions=function_schema,
            function_call={"name": "handleUserRequest"},
            temperature=0.0,
            top_p=1.0,
        )

        # Parse GPT's JSON
        arguments_str = response.choices[0].message.function_call.arguments
        data = json.loads(arguments_str)

        gpt_type = data.get("type", "chat")
        gpt_reply = data.get("reply", "")
        gpt_query = data.get("query", "")

        final_type = gpt_type
        final_query = gpt_query

        # Insert GPT's raw JSON in conversation
        raw_json_str = json.dumps(data, ensure_ascii=False)
        db_utils.insert_message(chat_req.sessionId, "assistant", raw_json_str)
        conversation.append({"role":"assistant","content":raw_json_str})

        # ========== CASE A: type='chat' ==========
        if gpt_type == "chat":
            # We unify final_message as the "reply"
            final_message = gpt_reply
            done = True
            break

        # ========== CASE B: type='sql' ==========
        elif gpt_type == "sql" and gpt_query.strip():
            # read-only check
            if not utils.is_read_only_sql(gpt_query):
                final_message = "Error: Attempted non-read-only query."
                final_type = "chat"
                done = True
                break

            # run the query
            db_results = utils.execute_sql(gpt_query)
            final_results = db_results

            # generate plain-language summary
            final_report = utils.generate_plain_report(gpt_query, db_results)

            # unify them into a single final_message for *this turn*
            partial_json = {
                "reply": gpt_reply,
                "query": gpt_query,
                "results": db_results,
                "final_report": final_report
            }
            merged_message = utils.merge_final_output_with_json_mode_multi_turn(partial_json)

            final_message = merged_message

            # save that final_message as GPT's assistant message
            db_utils.insert_message(chat_req.sessionId, "assistant", final_message)
            conversation.append({"role":"assistant","content":final_message})

            # store this turn's query & results in sql_history
            sql_history.append({
                "turn": turn_count,
                "query": gpt_query,
                "results": db_results,
                "merged_message": merged_message
            })

            # also insert the DB results as a system message so GPT sees them next turn
            results_str = json.dumps({"query_results": db_results}, ensure_ascii=False)
            db_utils.insert_message(chat_req.sessionId, "system", results_str)
            conversation.append({"role":"system", "content":results_str})

            # not done => next iteration
            continue

        # ========== CASE C: type='done' ==========
        elif gpt_type == "done":
            final_message = gpt_reply
            done = True
            break

        else:
            final_message = "Got an unexpected response from GPT."
            done = True

    # end while

    # return a single consistent message plus the entire sql_history
    return {
        "type": final_type,           # 'chat','sql','done'
        "final_message": final_message,
        "last_query": final_query,    # last query if any
        "last_results": final_results,# last results if any
        "sql_history": sql_history,   # all queries & results
        "turns_executed": turn_count
    }

def build_integrated_system_prompt() -> str:
    """
    Returns the big system prompt that handles multi-turn chat and SQL generation with self-critique.
    Allows type='chat', 'sql', or 'done'.
    """
    return (
        "You are 'Bo', a friendly, helpful assistant with read-only access to a SQLite database.\n\n"

        "You can only produce SQL queries that retrieve (SELECT) data—never modifying or deleting data. "
        "When the user requests data from the database, you must generate a valid single SQL query, referencing "
        "You always respond in the same language the user uses. If the user writes in Turkish, respond in Turkish. "
        "If they write in English, respond in English, etc.\n\n"  
        "only the schema below. Use SELECT, JOIN, WHERE, GROUP BY, ORDER BY, etc. for read-only. "
        "No INSERT, UPDATE, DELETE, DROP, or CREATE.\n\n"

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
    """
    GPT function schema for returning {type, reply, query}, allowing 'chat', 'sql', or 'done'.
    This is appropriate for a multi-turn endpoint where:
      - 'chat' means only a conversation reply (no query),
      - 'sql' means a valid single read-only SQL statement,
      - 'done' means no more queries or conversation are needed.
    """
    return [
        {
            "name": "handleUserRequest",
            "description": (
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
                            "A valid read-only SQL query if type='sql'. Otherwise, an empty string."
                        )
                    }
                },
                "required": ["type", "reply", "query"],
                "additionalProperties": False
            }
        }
    ]


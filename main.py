import os
from typing import Any, Dict, List
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import utils
import openai

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

@app.post("/final-report")
def final_report_endpoint(query_request: QueryRequest) -> Dict[str, Any]:
    """
    Converts a natural language query into a SQL query, executes it,
    and returns a concise final report based on the results.
    
    Args:
        query_request (QueryRequest): Contains the natural language query.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "sql": The generated SQL query,
            - "results": The SQL query results,
            - "final_report": A concise analysis of those results.
    """

    sql_output = utils.generate_sql_query_in_json(query_request.query)
    sql_query = sql_output.get("sql")

    sql_results = utils.execute_sql(sql_query)

    final_report = utils.generate_final_report(sql_query, sql_results)

    return {
        "sql": sql_query,
        "results": sql_results,
        "final_report": final_report
    }


@app.post("/decide")
def decide_context(user_message: UserMessage) -> Dict[str, Any]:
    """
    Endpoint that lets GPT decide the context:
    - If user wants general chat, returns {"type": "chat", "reply": "..."}.
    - If user wants SQL data, returns {"type": "sql", "query": "..."}.
    """

    system_prompt = utils.build_system_prompt_for_intention()


    functions_schema = utils.build_functions_schema()


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message.message},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=functions_schema,
        function_call={"name": "determineAction"},
    )

    parsed = utils.parse_function_arguments(response)

    if parsed["type"] == "chat":
        return {
            "type": "chat",
            "reply": parsed.get("reply", "")
        } 
    return {
        "type": "sql",
        "query": parsed.get("query", "")
    }
    

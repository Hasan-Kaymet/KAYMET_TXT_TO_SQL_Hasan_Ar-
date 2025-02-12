import os
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import utils

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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


@app.post("/generate-sql", response_model=Dict[str, str])
async def generate_sql(
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
    sql_query = await utils.generate_sql_query(query_request.query)
    if sql_query.startswith("-- Error"):
        raise HTTPException(status_code=500, detail=sql_query)
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
    results = await utils.execute_sql(sql_request.sql)
    if results and "error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["error"])
    return results

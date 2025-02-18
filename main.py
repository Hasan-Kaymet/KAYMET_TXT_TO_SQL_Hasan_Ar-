import os
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import utils

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


class QueryRequest(BaseModel):
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
    """
    sql_query = utils.generate_sql_query(query_request.query)
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
    """
    results = utils.execute_sql(sql_request.sql)
    if results and "error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["error"])
    return results




@app.post("/final-report", response_model=Dict[str, Any])
def final_report_endpoint(
    query_request: QueryRequest = Body(..., description="Natural language query.")
) -> Dict[str, Any]:
    """
    This endpoint receives a natural language query, generates a SQL query, executes it, 
    and produces a final detailed analysis report based on the SQL query and its results.
    """
    
    sql_output = utils.generate_sql_query_in_json(query_request.query)
    sql_query = sql_output.get("sql")

    
    sql_results = utils.execute_sql(sql_query)

    
    final_report = utils.generate_final_report(sql_query, sql_results)

    final_output = {
        "sql": sql_query,
        "results": sql_results,
        "final_report": final_report
    }
    return final_output

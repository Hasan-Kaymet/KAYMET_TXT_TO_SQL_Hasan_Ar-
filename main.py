from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils

app = FastAPI()


class RequestQuery(BaseModel):
    query: str


class SQLRequest(BaseModel):
    sql: str


@app.post("/generate-sql", response_model=Dict[str, str])
async def generate_sql(query_request: RequestQuery) -> Dict[str, str]:
    
    sql_query = utils.generate_sql_query(query_request.query)
    if sql_query.startswith("-- Error"):
        raise HTTPException(status_code=500, detail=sql_query)
    return {"sql": sql_query}


@app.post("/execute-sql", response_model=List[Dict[str, Any]])
async def execute_sql_endpoint(sql_request: SQLRequest) -> List[Dict[str, Any]]:
    
    results = utils.execute_sql(sql_request.sql)
    
    if results and "error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["error"])
    return results

# Chat & SQL Assistant

This project is a FastAPI-based application that integrates GPT (using OpenAI's API) with a SQLite database to provide interactive chat and SQL query functionality. The app handles natural language queries and generates corresponding SQL queries to retrieve data, along with plain-language summaries of the results.

## Features

- **Natural Language to SQL Conversion:**  
  Convert user-friendly natural language queries into valid, optimized SQL queries using GPT.
  
- **Interactive Chat:**  
  Engage in multi-turn conversations that combine regular chat with data-driven queries.
  
- **Database Integration:**  
  Execute generated SQL queries against a SQLite database to retrieve and present data.
  
- **Plain Language Reporting:**  
  Generate concise, non-technical reports from SQL query results.

## How It Works

1. **User Request Processing:**  
   Users send a request through one of the API endpoints (e.g., `/chat` or `/generate-sql`). The request can either be a general conversation or a query that requires data retrieval.

2. **Prompt Generation & GPT Integration:**  
   Depending on the request, the app builds a specialized system prompt that instructs GPT on whether to provide a conversational response or generate a SQL query. These prompts include detailed schema information and rules to ensure accurate output.

3. **SQL Execution:**  
   If a SQL query is generated, it is first checked to ensure it is read-only. The app then executes the SQL query against a SQLite database and retrieves the results.

4. **Plain Language Report:**  
   Optionally, the app can produce a plain language report that summarizes the SQL query results in a user-friendly manner without exposing technical details.

5. **Multi-turn Conversations:**  
   For multi-turn interactions, the app maintains a conversation history, enabling follow-up queries and iterative refinement of responses.


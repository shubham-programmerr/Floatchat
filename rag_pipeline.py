# rag_pipeline.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json

# --- Configuration ---
DB_CONNECTION_STRING = st.secrets["connections"]["postgres"]["url"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)
db_engine = create_engine(DB_CONNECTION_STRING)

@st.cache_resource
def get_retriever_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retriever_model = get_retriever_model()

# --- FAISS Vector Store ---
metadata_docs = [
    "Table 'argo_profiles' contains oceanographic data from ARGO floats.",
    "Column 'n_prof' is the unique identifier for each profile (measurement cycle).",
    "Column 'latitude' and 'longitude' are the float's coordinates.",
    "Column 'timestamp' is the date and time of the measurement.",
    "Column 'pressure' corresponds to depth in decibars.",
    "Column 'temperature' is the water temperature in degrees Celsius.",
    "Column 'salinity' is the practical salinity of the water."
]
doc_embeddings = retriever_model.encode(metadata_docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings.astype('float32'))

# --- Process User Question ---
def process_user_question(_question: str) -> dict:
    question_embedding = retriever_model.encode([_question])
    distances, indices = index.search(question_embedding.astype('float32'), k=4)
    context = "\n".join([metadata_docs[i] for i in indices[0]])

    template = """
    You are an expert PostgreSQL data scientist. Your task is to analyze a user's question about ARGO float data and return a JSON object with two fields: "sql_query" and "visualization_types".

    GENERAL RULES:
    1.  **sql_query**: Write a single, valid PostgreSQL query.
    2.  **visualization_types**: A list of strings: ["plot"], ["map"], or [].
    3.  **CRITICAL RULE: Only include "plot" in visualization_types if the user explicitly asks for a "plot", "graph", "chart", or "visualize". Simple "show" or "what is" questions should NOT generate a plot.**
    4.  **Map Plotting**: If the user asks for a "map" or "path", you MUST include the `n_prof`, `latitude`, and `longitude` columns in the SELECT statement for labeling.
    5.  **Prioritize User Numbers**: Always use the specific profile numbers or ranges the user provides. For "profiles 10 and 20", use `WHERE n_prof IN (10, 20)`. For "profiles 10 to 20", use `WHERE n_prof BETWEEN 10 AND 20`.
    6.  **Multi-Profile Plotting**: If a plot involves multiple profiles, you MUST include the `n_prof` column in the SELECT statement.
    7.  **JSON Only**: Return ONLY the JSON object, with no other text.

    ### Schema:
    CREATE TABLE argo_profiles (n_prof INTEGER, latitude FLOAT, longitude FLOAT, timestamp TIMESTAMP, pressure FLOAT, temperature FLOAT, salinity FLOAT, geometry GEOMETRY(Point, 4326));

    ### Context:
    {context}
    
    ### Few-Shot Examples:

    User Question: "Show the temperature for profiles 1 to 5."
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, temperature FROM argo_profiles WHERE n_prof BETWEEN 1 AND 5;",
        "visualization_types": []
    }}

    User Question: "Map the path of the float for the first 10 profiles."
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, latitude, longitude FROM argo_profiles WHERE n_prof <= 10;",
        "visualization_types": ["map"]
    }}

    User Question: "Plot the salinity vs pressure for profiles 1 through 5."
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, salinity, pressure FROM argo_profiles WHERE n_prof BETWEEN 1 AND 5;",
        "visualization_types": ["plot"]
    }}

    ### User Question:
    {question}

    ### JSON Response:
    """
    prompt = PromptTemplate.from_template(template).format(
        context=context,
        question=_question
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return {"error": "Failed to parse AI response as JSON."}
        
        json_str = json_match.group(0)
        ai_json = json.loads(json_str)
        
        sql_query = ai_json.get("sql_query", "").strip()
        sql_query = re.sub(r"```sql|```", "", sql_query, flags=re.IGNORECASE).strip()
        select_pos = sql_query.upper().find("SELECT")
        if select_pos != -1:
            sql_query = sql_query[select_pos:]
        ai_json["sql_query"] = sql_query

        return ai_json

    except (json.JSONDecodeError, Exception) as e:
        return {"error": f"An error occurred while processing the AI response: {str(e)}"}


# --- Execute SQL Safely ---
def execute_query(sql: str):
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return None, "❌ Only SELECT queries are allowed for safety."

    try:
        with db_engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn), None
    except Exception as e:
        return None, str(e)
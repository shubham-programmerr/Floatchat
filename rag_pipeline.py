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
# UPDATED: Added context for the new float_id column
metadata_docs = [
    "Table 'argo_profiles' contains oceanographic data from ARGO floats.",
    "Column 'float_id' is the identifier for the specific ARGO float.",
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

    # UPDATED: Prompt now includes the float_id column in the schema and examples
    template = """
    You are an expert PostgreSQL data scientist. Your task is to analyze a user's question about ARGO float data and return a JSON object with "sql_query" and "visualization_types".

    RULES:
    1.  **sql_query**: Write a single, valid PostgreSQL query. You MUST filter by the `float_id` when the user specifies one in their question.
    2.  **visualization_types**: A list of strings: ["plot"], ["map"], ["plot", "map"], or [].
    3.  If a user asks to "plot" or "graph" something but doesn't specify profiles, get the data for the first 50 measurements (ORDER BY timestamp LIMIT 50).
    4.  If the user asks for a map, YOU MUST include `n_prof`, `latitude`, and `longitude` in the SELECT statement.
    5.  When a user asks for a plot of multiple profiles, YOU MUST include `n_prof` in the SELECT statement.
    6.  Return ONLY the JSON object.

    ### Schema:
    CREATE TABLE argo_profiles (float_id TEXT, n_prof INTEGER, latitude FLOAT, longitude FLOAT, timestamp TIMESTAMP, pressure FLOAT, temperature FLOAT, salinity FLOAT, geometry GEOMETRY(Point, 4326));

    ### Context:
    {context}
    
    ### Few-Shot Examples:
    User Question: "Map the path for the first 10 profiles for float 1902671"
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, latitude, longitude FROM argo_profiles WHERE float_id = '1902671' AND n_prof <= 10;",
        "visualization_types": ["map"]
    }}

    User Question: "Plot the salinity vs pressure for profiles 1 through 5 for float 2902205."
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, salinity, pressure FROM argo_profiles WHERE float_id = '2902205' AND n_prof BETWEEN 1 AND 5;",
        "visualization_types": ["plot"]
    }}
    
    User Question: "Compare the plot for temperature and pressure for profiles 10 and 20 for float 6903240"
    JSON Response:
    {{
        "sql_query": "SELECT n_prof, temperature, pressure FROM argo_profiles WHERE float_id = '6903240' AND n_prof IN (10, 20);",
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

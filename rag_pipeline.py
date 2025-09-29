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

# --- FAISS Vector Store (Unchanged) ---
metadata_docs = [
    "Table 'argo_profiles' contains oceanographic data from ARGO floats.",
    "Column 'n_prof' is the unique identifier for each profile (measurement cycle).",
    "Column 'latitude' and 'longitude' are the float's coordinates.",
    "Column 'timestamp' is the date and time of the measurement. Higher timestamp values are more recent.",
    "Column 'pressure' corresponds to depth in decibars; higher pressure values mean deeper in the ocean.",
    "Column 'temperature' is the water temperature in degrees Celsius.",
    "Column 'salinity' is the practical salinity of the water."
]
doc_embeddings = retriever_model.encode(metadata_docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings.astype('float32'))

# --- UPDATED: This function now processes the user's intent, not just SQL ---
def process_user_question(question: str) -> dict:
    """
    Analyzes the user's question to generate SQL and determine if a visualization is requested.
    Returns a dictionary with 'sql_query' and 'visualization_requested' keys.
    """
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(question_embedding.astype('float32'), k=3)
    context = "\n".join([metadata_docs[i] for i in indices[0]])

    # --- UPDATED: The prompt now asks for a JSON object with two outputs ---
    template = """
    You are an expert PostgreSQL data scientist and a helpful assistant.
    Your task is to analyze the user's question and generate a valid PostgreSQL query. You must also determine if the user's question implies a request for a visualization (like a graph, chart, or map).

    Respond with a single, valid JSON object with two keys:
    1. "sql_query": A string containing the SQL query to answer the data part of the question.
    2. "visualization_requested": A boolean value (true or false). This should be true only if the user explicitly uses words like "graph", "plot", "chart", "visualize", "map", etc.

    - The JSON object must be the only thing you output. Do not add explanations or markdown.

    ### Schema:
    CREATE TABLE argo_profiles (n_prof INTEGER, latitude FLOAT, longitude FLOAT, timestamp TIMESTAMP, pressure FLOAT, temperature FLOAT, salinity FLOAT, geometry GEOMETRY(Point, 4326));

    ### Context:
    {context}

    ### Examples:
    User Question: "Show me the data for the most recent profile."
    JSON Response: {{"sql_query": "SELECT * FROM argo_profiles ORDER BY timestamp DESC LIMIT 1;", "visualization_requested": false}}

    User Question: "Can you plot the temperature against pressure for the first 5 profiles?"
    JSON Response: {{"sql_query": "SELECT n_prof, temperature, pressure FROM argo_profiles WHERE n_prof <= 5;", "visualization_requested": true}}

    ### User Question:
    {question}

    ### JSON Response:
    """
    prompt = PromptTemplate.from_template(template).format(
        current_date=pd.to_datetime('today').strftime('%Y-%m-%d'),
        context=context,
        question=question
    )
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # --- UPDATED: Clean and parse the JSON response from the AI ---
    response_text = re.sub(r"```json|```", "", response_text, flags=re.IGNORECASE).strip()
    
    try:
        result = json.loads(response_text)
        # Clean the nested SQL query
        if 'sql_query' in result and isinstance(result['sql_query'], str):
            result['sql_query'] = re.sub(r"```sql|```", "", result['sql_query'], flags=re.IGNORECASE).strip()
        return result
    except (json.JSONDecodeError, TypeError):
        # Fallback if the AI doesn't return valid JSON
        return {"sql_query": "SELECT 'AI response error: Could not parse JSON';", "visualization_requested": False, "error": f"Failed to decode AI JSON response: {response_text}"}


def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame or an error message."""
    if not sql.strip().upper().startswith("SELECT"):
        return None, "❌ Only SELECT queries are allowed for safety."

    try:
        with db_engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn), None
    except Exception as e:
        return None, str(e)
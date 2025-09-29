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
    groq_api_key=GROQ_API_KEY,
    temperature=0, # Set to 0 for deterministic JSON output
    model_kwargs={"response_format": {"type": "json_object"}},
)
db_engine = create_engine(DB_CONNECTION_STRING)

@st.cache_resource
def get_retriever_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retriever_model = get_retriever_model()

# --- Build the In-Memory FAISS Vector Store ---
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

# --- Process User Question (AI Core) ---
def process_user_question(question: str) -> dict:
    """
    Analyzes a user's question to generate a SQL query and determine the
    specific type of visualization requested (plot, map, or none).
    """
    question_embedding = retriever_model.encode([question])
    _, indices = index.search(question_embedding.astype('float32'), k=3)
    context = "\n".join([metadata_docs[i] for i in indices[0]])

    template = """
    You are an expert PostgreSQL data analyst. Your task is to analyze a user's question about ARGO ocean data and return a JSON object with two keys: "sql_query" and "visualization_types".

    1.  **sql_query**: Write a single, valid PostgreSQL query to answer the user's question.
    2.  **visualization_types**: A list of strings specifying what to visualize.
        - If the user asks to "plot", "chart", "visualize", or "show" data with two or more numerical variables (like temperature and pressure), include "plot".
        - If the user asks to "map" or see a "path" or "trajectory", include "map".
        - If the user only asks for data without a visual request, the list should be empty, like [].

    - Output ONLY the JSON object. Do not include markdown, code fences, or explanations.

    ### Schema:
    CREATE TABLE argo_profiles (
        n_prof INTEGER, 
        latitude FLOAT, 
        longitude FLOAT, 
        timestamp TIMESTAMP, 
        pressure FLOAT, 
        temperature FLOAT, 
        salinity FLOAT, 
        geometry GEOMETRY(Point, 4326)
    );

    ### Context:
    {context}
    
    ### Examples:
    User Question: "Show me the temperature vs pressure for the most recent 5 profiles."
    Your Response:
    {{
        "sql_query": "SELECT n_prof, temperature, pressure FROM argo_profiles ORDER BY timestamp DESC, n_prof DESC LIMIT 5;",
        "visualization_types": ["plot"]
    }}
    
    User Question: "Map the path of the float for the first 10 profiles."
    Your Response:
    {{
        "sql_query": "SELECT latitude, longitude FROM argo_profiles WHERE n_prof <= 10 ORDER BY n_prof;",
        "visualization_types": ["map"]
    }}

    User Question: "What is the average temperature?"
    Your Response:
    {{
        "sql_query": "SELECT AVG(temperature) as average_temperature FROM argo_profiles;",
        "visualization_types": []
    }}

    ### User Question:
    {question}

    ### Your Response (JSON only):
    """
    prompt = PromptTemplate.from_template(template).format(
        context=context,
        question=question
    )
    
    try:
        response = llm.invoke(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return {"error": str(e), "sql_query": None, "visualization_types": []}

# --- Execute SQL Safely ---
def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame and an error message."""
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return None, "❌ Only SELECT queries are allowed for safety."

    try:
        with db_engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn), None
    except Exception as e:
        return None, str(e)
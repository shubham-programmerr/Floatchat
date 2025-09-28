# rag_pipeline.py (Updated with Caching and Timeout)
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# --- Configuration ---
DB_CONNECTION_STRING = st.secrets["connections"]["postgres"]["url"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- Initialize Models and Database Connection ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# UPDATED: Added a 15-second statement timeout to the database connection
db_engine = create_engine(
    DB_CONNECTION_STRING,
    connect_args={"options": "-c statement_timeout=15000"}
)

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

# --- Generate SQL from Natural Language ---
# UPDATED: Added caching to prevent redundant API calls for the same question
@st.cache_data
def get_sql_from_question(question: str) -> str:
    """Generates and robustly cleans an SQL query from a natural language question."""
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(question_embedding.astype('float32'), k=3)
    context = "\n".join([metadata_docs[i] for i in indices[0]])

    template = """
    You are an expert PostgreSQL and PostGIS data scientist. 
    Given the table schema and context, write a single, valid SQL query to answer the user's question.
    - Do not include markdown, code fences, or explanations.
    - Only output the SQL query.
    - Use table and column names exactly as defined.

    The current date is {current_date}.

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

    ### User Question:
    {question}

    ### SQL Query:
    """
    prompt = PromptTemplate.from_template(template).format(
        current_date=pd.to_datetime('today').strftime('%Y-%m-%d'),
        context=context,
        question=question
    )
    
    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    # --- Clean up the output ---
    sql_query = re.sub(r"```sql|```", "", sql_query, flags=re.IGNORECASE).strip()

    select_pos = sql_query.upper().find("SELECT")
    if select_pos != -1:
        sql_query = sql_query[select_pos:]
        
    return sql_query.strip()

# --- Execute SQL Safely ---
def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame and an error message."""
    if not sql.strip().upper().startswith("SELECT"):
        return None, "❌ Only SELECT queries are allowed for safety."

    try:
        with db_engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn), None
    except Exception as e:
        return None, str(e)
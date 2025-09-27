# rag_pipeline.py (Definitive Version with Stable Model)
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
DB_CONNECTION_STRING = st.secrets["connections"]["postgres"]["url"]
GEMINI_API_KEY = st.secrets

# --- Initialize Models and Database Connection ---
# Use the stable, universally available model name
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GEMINI_API_KEY)
db_engine = create_engine(DB_CONNECTION_STRING)

@st.cache_resource
def get_retriever_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retriever_model = get_retriever_model()

# --- Build the In-Memory FAISS Vector Store ---
metadata_docs =
doc_embeddings = retriever_model.encode(metadata_docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings.astype('float32'))

def get_sql_from_question(question: str) -> str:
    """Generates and robustly cleans an SQL query from a natural language question."""
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(question_embedding.astype('float32'), k=3)
    context = "\n".join([metadata_docs[i] for i in indices])

    template = """
    You are an expert PostgreSQL and PostGIS data scientist. Given the table schema and context, write a single, valid SQL query to answer the user's question.
    The current date is {current_date}. Output ONLY the SQL query.

    ### Schema:
    CREATE TABLE argo_profiles (n_prof INTEGER, latitude FLOAT, longitude FLOAT, timestamp TIMESTAMP, pressure FLOAT, temperature FLOAT, salinity FLOAT, doxy_adjusted FLOAT, chla_adjusted FLOAT, geometry GEOMETRY(Point, 4326));

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
    
    # Robustly find the start of the SQL command
    select_pos = sql_query.upper().find("SELECT")
    if select_pos!= -1:
        sql_query = sql_query[select_pos:]
    
    # Remove trailing markdown backticks
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
        
    return sql_query.strip()

def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame and error message."""
    try:
        return pd.read_sql_query(sql, db_engine), None
    except Exception as e:
        return None, str(e)
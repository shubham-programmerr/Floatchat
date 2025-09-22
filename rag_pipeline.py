# rag_pipeline.py (Final Version with FAISS)
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
GOOGLE_API_key = st.secrets["AIzaSyBEpSu05Z3jeroEHB4eti15DCSEJfovqc0"]

# --- Initialize Models and Database Connection ---
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_key)
db_engine = create_engine(DB_CONNECTION_STRING)
# Use a cached model for sentence embeddings
@st.cache_resource
def get_retriever_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retriever_model = get_retriever_model()

# --- Build the In-Memory FAISS Vector Store ---
metadata_docs = [
    "Table 'argo_profiles' contains oceanographic data from ARGO floats.",
    "Column 'latitude' and 'longitude' are the float's coordinates.",
    "Column 'timestamp' is the date and time of the measurement.",
    "Column 'pressure' corresponds to depth in decibars.",
    "Column 'temperature' is the water temperature in degrees Celsius.",
    "Column 'salinity' is the practical salinity of the water.",
    "Column 'doxy_adjusted' is the dissolved oxygen level.",
    "Column 'chla_adjusted' is the Chlorophyll-a concentration."
]

doc_embeddings = retriever_model.encode(metadata_docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings.astype('float32'))

def get_sql_from_question(question: str) -> str:
    """Generates an SQL query from a natural language question using FAISS for RAG."""
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(question_embedding.astype('float32'), k=3)
    
    context = "\n".join([metadata_docs[i] for i in indices[0]])

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
    return response.content.strip()

def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame and error message."""
    try:
        return pd.read_sql_query(sql, db_engine), None
    except Exception as e:
        return None, str(e)
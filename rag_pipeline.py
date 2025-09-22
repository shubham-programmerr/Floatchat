# rag_pipeline.py
import streamlit as st
import chromadb
import pandas as pd
from sqlalchemy import create_engine
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Use Streamlit's secrets for database connection and API key
DB_CONNECTION_STRING = st.secrets["connections"]["postgres"]["url"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize connections
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
db_engine = create_engine(DB_CONNECTION_STRING)

# Initialize ChromaDB and populate with metadata
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
client = chromadb.Client()
collection = client.get_or_create_collection("argo_schema_docs")
if collection.count() == 0:
    collection.add(
        documents=metadata_docs,
        ids=[f"doc_{i}" for i in range(len(metadata_docs))]
    )

def get_sql_from_question(question: str) -> str:
    """Generates a SQL query from a natural language question using RAG."""
    retrieved_docs = collection.query(query_texts=[question], n_results=4)
    context = "\n".join(retrieved_docs['documents'][0])
    
    template = """
    You are an expert PostgreSQL and PostGIS data scientist. Given the table schema and context, write a single, valid SQL query to answer the user's question.
    Use date functions for time queries. The current date is {current_date}.
    Use PostGIS functions for geospatial queries. Output ONLY the SQL query.

    ### Schema:
    CREATE TABLE argo_profiles (N_PROF INTEGER, latitude FLOAT, longitude FLOAT, timestamp TIMESTAMP, pressure FLOAT, temperature FLOAT, salinity FLOAT, doxy_adjusted FLOAT, chla_adjusted FLOAT, geometry GEOMETRY(Point, 4326));

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
    # Using .invoke for newer LangChain versions with Gemini
    response = llm.invoke(prompt)
    return response.content.strip()

def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame and error message."""
    try:
        return pd.read_sql_query(sql, db_engine), None
    except Exception as e:
        return None, str(e)
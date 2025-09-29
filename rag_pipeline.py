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

# ----------------------------------------------------
# 1. Configuration
# ----------------------------------------------------
DB_CONNECTION_STRING = st.secrets["connections"]["postgres"]["url"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # New secret name
llm = ChatGroq(
    model="llama3-70b-8192", # Using a Llama 3 model on Groq
    groq_api_key=GROQ_API_KEY
)

# ----------------------------------------------------
# 3. Initialize Database
# ----------------------------------------------------
db_engine = create_engine(DB_CONNECTION_STRING)

# ----------------------------------------------------
# 4. Initialize FAISS retriever
# ----------------------------------------------------
@st.cache_resource
def get_retriever_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retriever_model = get_retriever_model()

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

# ----------------------------------------------------
# 5. Generate SQL from natural language
# ----------------------------------------------------
def get_sql_from_question(question: str) -> str:
    """Generates a cleaned SQL query from a natural language question."""
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(question_embedding.astype('float32'), k=3)
    context = "\n".join([metadata_docs[i] for i in indices[0]])

    template = """
    You are an expert PostgreSQL and PostGIS data scientist. 
    Given the table schema and context, write a single, valid SQL query to answer the user's question.
    - Only output the SQL query. No markdown or explanations.
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
    
    # Call the LLM
    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    # Remove any markdown code fences
    sql_query = re.sub(r"```sql|```", "", sql_query, flags=re.IGNORECASE).strip()

    # Ensure query starts with SELECT
    select_pos = sql_query.upper().find("SELECT")
    if select_pos != -1:
        sql_query = sql_query[select_pos:]

    return sql_query.strip()

# ----------------------------------------------------
# 6. Execute SQL safely
# ----------------------------------------------------
def execute_query(sql: str):
    """Executes the SQL query and returns a DataFrame or an error message."""
    if not sql.strip().upper().startswith("SELECT"):
        return None, "❌ Only SELECT queries are allowed for safety."

    try:
        with db_engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn), None
    except Exception as e:
        return None, str(e)

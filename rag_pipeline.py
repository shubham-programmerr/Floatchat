# app.py (Temporary Debugging Version)
import streamlit as st

st.set_page_config(page_title="Secrets Debugger")
st.title("Secrets Debugger")

st.info("This app will check the secrets available in the Streamlit Cloud environment.")

# Check if secrets are loaded at all
if "connections" in st.secrets and "postgres" in st.secrets["connections"]:
    st.success("Found [connections.postgres] secret!")
else:
    st.error("Did not find [connections.postgres] secret.")

if "GEMINI_API_KEY" in st.secrets:
    st.success("Found GEMINI_API_KEY secret!")
    st.write("First 4 characters of key:", st.secrets["GEMINI_API_KEY"][:4])
else:
    st.error("Did not find GEMINI_API_KEY secret.")

st.warning("Below is the full dictionary of all secrets found:")
st.write(st.secrets.to_dict())
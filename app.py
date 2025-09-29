# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from rag_pipeline import process_user_question, execute_query

# --- Page Configuration ---
st.set_page_config(
    page_title="ProCode FloatChat",
    page_icon="🌊",
    layout="wide"
)

# --- Custom CSS for a better UI ---
st.markdown("""
<style>
    /* Main app styling for dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Ensure title is visible */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa;
    }
    /* Chat bubble styling */
    .st-chat-message-container {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Assistant message styling */
    [data-testid="stChatMessage"][data-testid="stChatMessageContent"] {
        background-color: #262730;
        color: #fafafa;
    }
    /* User message styling */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContentUser"]) {
        background-color: #1c3346;
        color: #fafafa;
    }
    /* Expander styling */
    .st-expander {
        border: 1px solid #31333F;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🌊 FloatChat")
    st.markdown("An AI-powered conversational interface for exploring ARGO ocean data. Ask questions in natural language and get back data, charts, and maps.")
    st.markdown("---")
    st.header("Example Prompts")
    
    example_prompts = [
        "Show the temperature and pressure for the first 10 profiles.",
        "Plot the salinity vs pressure for profiles 1 through 5.",
        "Map the float's path for the first 50 profiles.",
        "What is the average temperature for each of the first 5 profiles?"
    ]
    
    # Create buttons for each example
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state.prefilled_prompt = prompt

# --- Main App Logic ---
st.title("FloatChat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the ARGO float data today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

# Handle chat input, pre-filling from sidebar buttons if necessary
user_prompt = st.chat_input("Ask about the ARGO data...", key="chat_input")
if st.session_state.get("prefilled_prompt"):
    user_prompt = st.session_state.prefilled_prompt
    # Clear the prefilled prompt so it's not reused
    del st.session_state.prefilled_prompt

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and querying the database..."):
            ai_response = process_user_question(user_prompt)
            sql_query = ai_response.get("sql_query")
            show_visuals = ai_response.get("visualization_requested", False)
            ai_error = ai_response.get("error")

            if ai_error:
                st.warning("Sorry, I had trouble understanding that. Could you please try rephrasing?")
                with st.expander("See technical details"):
                    st.error(ai_error)
                st.stop()
            
            if not sql_query:
                st.warning("I couldn't generate a query for that request. Please try again.")
                st.stop()

            with st.expander("🔍 Generated SQL Query", expanded=False):
                st.code(sql_query, language="sql")
            
            result_df, db_error = execute_query(sql_query)
            
            if db_error:
                st.warning("Sorry, there was a problem fetching data. This could be a temporary issue.")
                with st.expander("See technical details"):
                    st.error(db_error)
                st.stop()

            st.success("Query executed successfully!")
            st.dataframe(result_df, use_container_width=True)
            st.session_state.messages.append({"role": "assistant", "content": result_df})

            if not result_df.empty:
                if show_visuals:
                    with st.expander("📊 Visualizations & Export", expanded=True):
                        vis_col, export_col = st.columns([3, 1])
                        
                        with vis_col:
                            if 'latitude' in result_df.columns and 'longitude' in result_df.columns and result_df['latitude'].nunique() > 1:
                                st.caption("Float Trajectory/Positions")
                                st.map(result_df[['latitude', 'longitude']])
                            
                            if 'n_prof' in result_df.columns and result_df['n_prof'].nunique() > 1 and 'temperature' in result_df.columns and 'pressure' in result_df.columns:
                                st.caption("Profile Comparison")
                                fig = px.line(result_df, y='pressure', x='temperature', color='n_prof', title='Profile Comparison')
                                fig.update_yaxes(autorange="reversed")
                                st.plotly_chart(fig, use_container_width=True)

                        with export_col:
                            st.caption("Download Data")
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download as CSV", csv, "argo_data.csv", "text/csv", key='csv')
                            
                            required_cols = {'n_prof', 'pressure'}
                            if required_cols.issubset(result_df.columns):
                                try:
                                    df_for_export = result_df.drop(columns=['geometry'], errors='ignore')
                                    df_indexed = df_for_export.set_index(['n_prof', 'pressure'])
                                    ds_export = df_indexed.to_xarray()
                                    netcdf_bytes = ds_export.to_netcdf()
                                    st.download_button("Download as NetCDF", data=bytes(netcdf_bytes), file_name="argo_data.nc", mime="application/x-netcdf", key='netcdf')
                                except Exception as e:
                                    st.warning("Failed to generate NetCDF file.")
                                    with st.expander("See technical details"):
                                        st.error(e)
                else:
                    with st.expander("📊 Export", expanded=True):
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download as CSV", csv, "argo_data.csv", "text/csv", key='export_csv_no_viz')
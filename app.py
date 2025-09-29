# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from rag_pipeline import process_user_question, execute_query

# --- Page Configuration ---
st.set_page_config(
    page_title="ProCode-FloatChat",
    page_icon="🌊",
    layout="wide"
)

# --- Custom CSS for a better UI ---
st.markdown("""
<style>
    /* Main app styling for a black and white theme */
    .stApp {
        background-color: #121212; /* Dark background */
        color: #ffffff; /* White text */
    }
    h1, h2, h3, h4, h5, h6 { color: #ffffff; }
    .st-chat-message-container {
        border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;
        border: 1px solid #333333; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    [data-testid="stChatMessage"][data-testid="stChatMessageContent"] {
        background-color: #222222; color: #ffffff;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContentUser"]) {
        background-color: #333333; color: #ffffff;
    }
    .st-expander { border: 1px solid #444444; border-radius: 0.5rem; }
    [data-testid="stHeader"], [data-testid="stChatInputContainer"] {
        background-color: #121212;
    }
    [data-testid="stChatInputContainer"] { border-top: 1px solid #333333; }
    [data-testid="stChatInput"] { color: #ffffff; }
    [data-testid="stButton"] button {
        background-color: #ffffff; color: #121212;
        border: 1px solid #ffffff; border-radius: 0.5rem;
    }
    [data-testid="stButton"] button:hover {
        background-color: #dddddd; border-color: #dddddd;
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
    
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state.prefilled_prompt = prompt

# --- Main App Logic ---
st.title("FloatChat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the ARGO float data today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

user_prompt = st.chat_input("Ask about the ARGO data...", key="chat_input")
if st.session_state.get("prefilled_prompt"):
    user_prompt = st.session_state.prefilled_prompt
    del st.session_state.prefilled_prompt

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and querying the database..."):
            ai_response = process_user_question(user_prompt)
            sql_query = ai_response.get("sql_query")
            requested_visuals = ai_response.get("visualization_types", [])
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
                if requested_visuals:
                    with st.expander("📊 Visualizations & Export", expanded=True):
                        vis_col, export_col = st.columns([3, 1])
                        
                        with vis_col:
                            # --- Map Visualization ---
                            if "map" in requested_visuals and 'latitude' in result_df.columns and 'longitude' in result_df.columns:
                                st.caption("Float Trajectory/Positions")
                                st.map(result_df[['latitude', 'longitude']])
                            
                            # --- Plot Visualization (UPDATED with flexible logic) ---
                            if "plot" in requested_visuals:
                                st.caption("Profile Comparison")
                                numeric_cols = result_df.select_dtypes(include=np.number).columns.tolist()
                                
                                # Try to find standard axes, otherwise use the first two numeric columns
                                x_axis = 'temperature' if 'temperature' in numeric_cols else numeric_cols[0] if len(numeric_cols) > 0 else None
                                y_axis = 'pressure' if 'pressure' in numeric_cols else numeric_cols[1] if len(numeric_cols) > 1 else None
                                
                                if x_axis and y_axis:
                                    # Use n_prof for color if it exists, otherwise create a simple plot
                                    color_axis = 'n_prof' if 'n_prof' in result_df.columns else None
                                    
                                    fig = px.line(result_df, x=x_axis, y=y_axis, color=color_axis, title=f'{x_axis.capitalize()} vs. {y_axis.capitalize()}')
                                    
                                    # Invert y-axis if it's pressure
                                    if y_axis == 'pressure':
                                        fig.update_yaxes(autorange="reversed")
                                        
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Not enough data columns to generate a plot.")

                        with export_col:
                            st.caption("Download Data")
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download as CSV", csv, "argo_data.csv", "text/csv", key='csv')
                            
                            if 'n_prof' in result_df.columns and 'pressure' in result_df.columns:
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

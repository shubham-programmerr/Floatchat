# app.py (Final Version with Visualization Toggle)
import streamlit as st
import pandas as pd
import plotly.express as px
from rag_pipeline import get_sql_from_question, execute_query

# --- Page Configuration ---
st.set_page_config(page_title="FloatChat", layout="wide")
st.title("FloatChat 🌊 - AI Interface for ARGO Ocean Data")
st.caption("A Proof-of-Concept powered by Google Gemini")
st.markdown("Data sourced from the Argo Program, including the [Indian Argo Project](https://incois.gov.in/OON/index.jsp).")

# --- Main App Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Show the temperature and pressure for the first 10 profiles..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and querying the database..."):
            sql_query = get_sql_from_question(prompt)
            st.markdown("##### 🔍 Generated SQL Query:")
            st.code(sql_query, language="sql")
            
            result_df, error = execute_query(sql_query)
            
            if error:
                st.error(f"An error occurred: {error}")
            else:
                st.success("Query executed successfully!")
                st.dataframe(result_df, use_container_width=True)
                st.session_state.messages.append({"role": "assistant", "content": result_df})

                if not result_df.empty:
                    st.markdown("---")
                    st.subheader("📊 Visualizations & Export")
                    
                    # --- ADDED CHECKBOX FOR CONTROL ---
                    show_visuals = st.checkbox("Show Visualizations", value=True)
                    
                    vis_col, export_col = st.columns([3, 1])
                    
                    # --- VISUALS ARE NOW INSIDE THIS 'IF' BLOCK ---
                    if show_visuals:
                        with vis_col:
                            # Condition to show the map
                            if 'latitude' in result_df.columns and 'longitude' in result_df.columns and result_df['latitude'].nunique() > 1:
                                st.caption("Float Trajectory/Positions")
                                st.map(result_df[['latitude', 'longitude']])
                            
                            # Condition to show the comparison chart
                            if 'n_prof' in result_df.columns and result_df['n_prof'].nunique() > 1 and 'temperature' in result_df.columns and 'pressure' in result_df.columns:
                                st.caption("Profile Comparison")
                                fig = px.line(result_df, y='pressure', x='temperature', color='n_prof', title='Profile Comparison')
                                fig.update_yaxes(autorange="reversed")
                                st.plotly_chart(fig, use_container_width=True)

                    with export_col:
                        st.caption("Download Data")
                        
                        # CSV export
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download as CSV", csv, "argo_data.csv", "text/csv", key='csv')
                        
                        # NetCDF export
                        required_cols = {'n_prof', 'pressure'}
                        if required_cols.issubset(result_df.columns):
                            try:
                                df_for_export = result_df.drop(columns=['geometry'], errors='ignore')
                                df_indexed = df_for_export.set_index(['n_prof', 'pressure'])
                                ds_export = df_indexed.to_xarray()
                                netcdf_bytes = ds_export.to_netcdf()
                                
                                st.download_button(
                                    "Download as NetCDF", 
                                    netcdf_bytes, 
                                    "argo_data.nc", 
                                    "application/x-netcdf", 
                                    key='netcdf'
                                )
                            except Exception as e:
                                st.error(f"Failed to generate NetCDF file: {e}")
                        else:
                            st.warning("NetCDF export requires 'n_prof' and 'pressure' columns.")
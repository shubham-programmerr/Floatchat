# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
# CORRECTED: Import the new function name from the RAG pipeline
from rag_pipeline import process_user_question, execute_query

# --- Page Configuration ---
st.set_page_config(page_title="FloatChat", layout="wide")
st.title("FloatChat 🌊 - AI Interface for ARGO Ocean Data")
st.caption("A Proof-of-Concept powered by Groq and Llama 3.1")
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
            # CORRECTED: Call the new function and handle its dictionary output
            ai_response = process_user_question(prompt)
            sql_query = ai_response.get("sql_query")
            show_visuals = ai_response.get("visualization_requested", False)
            ai_error = ai_response.get("error")

            if ai_error:
                st.warning("Sorry, I had trouble understanding that. Could you please try rephrasing your question?")
                with st.expander("See technical details"):
                    st.error(ai_error)
                st.stop()
            
            if not sql_query:
                st.warning("I couldn't generate a query for that request. Please try asking in a different way.")
                st.stop()

            st.markdown("##### 🔍 Generated SQL Query:")
            st.code(sql_query, language="sql")
            
            result_df, db_error = execute_query(sql_query)
            
            if db_error:
                st.warning("Sorry, there was a problem fetching data from the database. This could be a temporary issue.")
                with st.expander("See technical details"):
                    st.error(db_error)
                st.stop()

            st.success("Query executed successfully!")
            st.dataframe(result_df, use_container_width=True)
            st.session_state.messages.append({"role": "assistant", "content": result_df})

            if not result_df.empty:
                st.markdown("---")
                
                if show_visuals:
                    st.subheader("📊 Visualizations & Export")
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
                                # CORRECTED: Convert memoryview to bytes to fix the error
                                st.download_button("Download as NetCDF", data=bytes(netcdf_bytes), file_name="argo_data.nc", mime="application/x-netcdf", key='netcdf')
                            except Exception as e:
                                st.warning("Sorry, failed to generate the NetCDF file.")
                                with st.expander("See technical details"):
                                    st.error(e)
                        else:
                            st.warning("NetCDF export requires 'n_prof' and 'pressure' columns.")
                else:
                    st.subheader("📊 Export")
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download as CSV", csv, "argo_data.csv", "text/csv", key='export_csv')
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pydeck as pdk
from rag_pipeline import process_user_question, execute_query

# --- Page Configuration ---
st.set_page_config(
    page_title="ProCode-FloatChat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded" 
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
    /* Main screen button styling */
    .stButton>button {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #444444;
        border-color: #555555;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🌊 FloatChat")
    st.markdown("An AI-powered conversational interface for exploring ARGO ocean data. Ask questions in natural language and get back data, charts, and maps.")
    st.markdown("---")
    
    # --- Input for specific float ID ---
    st.header("Select a Float")
    float_id = st.text_input("Enter ARGO Float ID:", value="1902671")
    
    st.markdown("---")
    st.info("This project was developed by **ProCode** for the Smart India Hackathon.")


# --- Main App Logic ---
st.title("FloatChat Interface")

# --- UPDATED: Description with specific float examples ---
st.info(
    f"ℹ️ **Note:** You are currently querying for float **{float_id}**. "
    "You can change the ID in the sidebar (e.g., try `1902671` or `5906266`)."
)

# --- UPDATED: Example Prompts now include the float_id context ---
st.markdown("##### Try an example prompt:")
example_prompts = [
    f"Show temperature and pressure for the first 10 profiles for float {float_id}",
    f"Plot salinity vs pressure for profiles 1 through 5 for float {float_id}",
    f"Map the float's path for the first 50 profiles for float {float_id}",
    f"What is the average temperature for each of the first 5 profiles for float {float_id}?"
]

# Create columns for the buttons
# We use the raw prompt for the button text to keep it clean
raw_prompts = [
    "Show the temperature and pressure for the first 10 profiles.",
    "Plot the salinity vs pressure for profiles 1 through 5.",
    "Map the float's path for the first 50 profiles.",
    "What is the average temperature for each of the first 5 profiles?"
]
cols = st.columns(len(raw_prompts))
for i, (raw_prompt, full_prompt) in enumerate(zip(raw_prompts, example_prompts)):
    with cols[i]:
        if st.button(raw_prompt, key=f"example_{i}"):
            # When button is clicked, use the full prompt with the float ID
            st.session_state.prefilled_prompt = full_prompt

st.markdown("---") # Add a separator

# --- Chat History and Input ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the ARGO float data today?"}]

# Display chat messages from history, but show the simplified user prompt
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # A bit of logic to display the raw prompt in the chat history
        content = message["content"]
        if isinstance(content, str) and f" for float {float_id}" in content:
            display_content = content.replace(f" for float {float_id}", "")
        else:
            display_content = content

        if isinstance(display_content, pd.DataFrame):
            st.dataframe(display_content, use_container_width=True)
        else:
            st.markdown(display_content)

user_prompt_from_input = st.chat_input("Ask about the ARGO data...", key="chat_input")
user_prompt_from_button = st.session_state.get("prefilled_prompt")

final_user_prompt = None
if user_prompt_from_input:
    # If user types, combine their text with the float_id
    final_user_prompt = f"{user_prompt_from_input} for float {float_id}"
    st.session_state.messages.append({"role": "user", "content": user_prompt_from_input})

elif user_prompt_from_button:
    # If user clicks button, use the full prompt from the button
    final_user_prompt = user_prompt_from_button
    simplified_prompt = user_prompt_from_button.replace(f" for float {float_id}", "")
    st.session_state.messages.append({"role": "user", "content": simplified_prompt})
    del st.session_state.prefilled_prompt


if final_user_prompt:
    # Rerun to show the user message immediately
    st.rerun()

# This block will now run on the rerun after a message is added
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and querying the database..."):
            # Use the last message content which is the full prompt
            last_user_message = st.session_state.messages[-1]["content"]
            full_prompt_to_process = f"{last_user_message} for float {float_id}"
            
            ai_response = process_user_question(full_prompt_to_process)
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
                            if "map" in requested_visuals:
                                if 'latitude' in result_df.columns and 'longitude' in result_df.columns and 'n_prof' in result_df.columns:
                                    st.caption("Float Positions (Hover for Profile ID)")
                                    
                                    map_df = result_df.sort_values(by='n_prof').copy()

                                    view_state = pdk.ViewState(
                                        latitude=map_df["latitude"].mean(),
                                        longitude=map_df["longitude"].mean(),
                                        zoom=7,
                                        pitch=0,
                                    )

                                    scatter_layer = pdk.Layer(
                                        "ScatterplotLayer",
                                        data=map_df,
                                        get_position="[longitude, latitude]",
                                        get_color="[255, 0, 0, 200]",
                                        get_radius=5000,
                                        pickable=True,
                                    )
                                    
                                    tooltip = {
                                        "html": "<b>Profile:</b> {n_prof}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
                                        "style": {"backgroundColor": "#333333", "color": "white", "border": "1px solid #444444"}
                                    }
                                    
                                    mapbox_key = st.secrets.get("MAPBOX_API_KEY")
                                    deck_kwargs = {
                                        "initial_view_state": view_state,
                                        "layers": [scatter_layer],
                                        "tooltip": tooltip
                                    }
                                    if mapbox_key:
                                        deck_kwargs["map_style"] = "mapbox://styles/mapbox/dark-v9"
                                        deck_kwargs["mapbox_key"] = mapbox_key
                                    
                                    st.pydeck_chart(pdk.Deck(**deck_kwargs))

                                else:
                                    st.warning("Could not generate a map. Query did not return 'n_prof', 'latitude', and 'longitude'.")
                            
                            if "plot" in requested_visuals:
                                st.caption("Data Plot")
                                numeric_cols = result_df.select_dtypes(include=np.number).columns.tolist()
                                color_col = 'n_prof' if 'n_prof' in result_df.columns else None
                                if color_col:
                                    numeric_cols.remove(color_col)
                                if len(numeric_cols) < 2:
                                    st.warning("Not enough data columns to generate a plot.")
                                else:
                                    y_axis = 'pressure' if 'pressure' in numeric_cols else numeric_cols[1]
                                    x_candidates = ['temperature', 'salinity']
                                    x_axis = next((col for col in x_candidates if col in numeric_cols), numeric_cols[0])
                                    df_to_plot = result_df.copy()
                                    if color_col:
                                        df_to_plot[color_col] = df_to_plot[color_col].astype(str)
                                    fig = px.line(df_to_plot, x=x_axis, y=y_axis, color=color_col, title=f'{x_axis.capitalize()} vs. {y_axis.capitalize()}')
                                    if y_axis == 'pressure':
                                        fig.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig, use_container_width=True)
                                    if color_col:
                                        st.info("💡 Tip: Double-click a profile in the legend to view it in isolation.")

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

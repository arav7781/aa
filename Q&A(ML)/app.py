import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["GROQ_API_KEY"] = "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"
import streamlit as st

# Set page configuration here, in the main app file
st.set_page_config(layout="wide", page_title="Educational Dashboard", page_icon="📚")

data_visualisation_page = st.Page(
    "C:\\Users\\aravs\\Desktop\\Q&A(ML)\\app_pages\\Prompts\\python_vizualization_agent.py", 
    title="Data Visualisation", 
    icon="📈"
)

pg = st.navigation(
    {
        "Visualisation Agent": [data_visualisation_page]
    }
)

pg.run()
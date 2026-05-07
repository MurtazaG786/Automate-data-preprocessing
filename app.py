import os
import streamlit as st
import streamlit.components.v1 as components
from workflow import graph

st.set_page_config(
    page_title="Automated Data Preprocessing",
    layout="wide"
)

st.title("Automated Data Preprocessing Agent")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    result = graph.invoke({
        "uploaded_file": uploaded_file
    })
    
    
    
    st.session_state["result"] = result
    if "result" in st.session_state:

        result = st.session_state["result"]
        st.write(result["message"])

    
    if result.get("error"):
        st.error(result["error"])

    else:
        if os.path.exists("ydata_report.html"):
            with open("ydata_report.html", "r", encoding="utf-8") as html_file:
                components.html(html_file.read(), height=800, scrolling=True)
        else:
            st.error("Report file not found: ydata_report.html")

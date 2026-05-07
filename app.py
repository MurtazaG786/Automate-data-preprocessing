import streamlit as st
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

if st.button("Run Agent"):
    result = graph.invoke({
        "uploaded_file": uploaded_file
    })

    st.session_state["result"] = result

    
if "result" in st.session_state:

    result = st.session_state["result"]

    if result.get("error"):
        st.error(result["error"])

    else:
        st.success(result["message"])

        st.write("### Dataset Shape")
        st.write(f"Rows: {result['rows']}")
        st.write(f"Columns: {result['cols']}")

        st.write("### Dataset Preview")
        st.dataframe(result["df"].head())
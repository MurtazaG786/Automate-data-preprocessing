import os
import tempfile
import uuid
import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command
from workflow import graph

st.set_page_config(
    page_title="Automated Data Preprocessing",
    layout="wide"
)

st.title("Automated Data Preprocessing Agent")

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp()

    temp_dir = st.session_state["temp_dir"]

    input_path = os.path.join(temp_dir, "input.csv")
    output_path = os.path.join(temp_dir, "processed.csv")
    report_path = os.path.join(temp_dir, "ydata_report.html")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "result" not in st.session_state:
        result = graph.invoke({
                "input_file_path":input_path,
                "output_file_path":output_path,
                "report_path":report_path,
                "steps":[],
                "numerical_pipeline":None,
                "categorical_pipeline":None,
                "final_preprocessor":None
            },

            config=config
        )

        st.session_state["result"] = result
        st.session_state["output_path"] = output_path
        st.session_state["report_path"] = report_path


if "result" in st.session_state:

    result = st.session_state["result"]

    steps = result.get("steps", [])

    if steps:
        st.subheader("Pipeline Progress")
        for step in steps:
            st.success(step)

    report_path = st.session_state.get("report_path")

    if report_path and os.path.exists(report_path):
        st.subheader("Dataset Report")
        with open(report_path, "r", encoding="utf-8") as html_file:
            components.html(
                html_file.read(),
                height=800,
                scrolling=True
            )

    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        st.info("Target column confirmation required")

        detected_target = interrupt_data.get("detected_target")
        confidence = interrupt_data.get("confidence")
        reason = interrupt_data.get("reason")
        columns = interrupt_data.get("columns", [])

        st.write(f"Detected Target: **{detected_target}**")
        st.write(f"Confidence: **{confidence}**")
        st.write(f"Reason: {reason}")

        selected_target = st.selectbox(
            "Select correct target column",
            options=["No target column"] + columns,
            index=(columns.index(detected_target) + 1)
            if detected_target in columns else 0
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Confirm Detected Target"):
                result = graph.invoke(
                    Command(resume={"approved": True}),
                    config=config
                )


            

                st.write("STATE AFTER RESUME")
                st.write(result)
                st.session_state["result"] = result
                st.rerun()

        with col2:
            if st.button("Use Selected / Unsupervised"):

                if selected_target == "No target column":
                    resume_data = {
                        "approved": False,
                        "target_column": None
                    }
                else:
                    resume_data = {
                        "approved": False,
                        "target_column": selected_target
                    }

                result = graph.invoke(
                    Command(resume=resume_data),
                    config=config
                )

                st.session_state["result"] = result
                st.rerun()

    else:

        if result.get("message"):
            st.success(result["message"])

        if result.get("problem_type"):
            st.write(f"Problem Type: **{result['problem_type']}**")

        if result.get("target_column"):
            st.write(f"Target Column: **{result['target_column']}**")

        if result.get("error"):
            st.error(result["error"])

        else:
            # ── Split Info ──────────────────────────────────────────
            train_rows = result.get("train_rows")
            test_rows = result.get("test_rows")

            if train_rows and test_rows:
                st.subheader("Train / Test Split")
                col_a, col_b = st.columns(2)
                col_a.metric("Training Rows", train_rows)
                col_b.metric("Testing Rows", test_rows)

            # ── Column Classification ───────────────────────────────
            num_cols = result.get("numerical_columns", [])
            cat_cols = result.get("categorical_columns", [])

            if num_cols or cat_cols:
                st.subheader("Column Classification")

                col_c, col_d = st.columns(2)

                with col_c:
                    st.markdown("**Numerical Columns**")
                    for c in num_cols:
                        st.write(f"- {c}")

                with col_d:
                    st.markdown("**Categorical Columns**")
                    for c in cat_cols:
                        st.write(f"- {c}")

            # -----------------------------
            # Downloads
            # -----------------------------

            st.subheader("Downloads")

            train_path = result.get(
                "train_path"
            )

            test_path = result.get(
                "test_path"
            )

            preprocessor_path = result.get(
                "final_preprocessor_path"
            )

            processed_train = result.get(
                "processed_train_path"
            )

            processed_test = result.get(
                "processed_test_path"
            )


            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )


            # -----------------------
            # Train CSV
            # -----------------------

            if train_path and os.path.exists(
                    train_path
            ):

                with open(
                    train_path,
                    "rb"
                ) as f:

                    col1.download_button(

                        "Train CSV",

                        data=f.read(),

                        file_name=
                        "train.csv",

                        mime=
                        "text/csv"
                    )


            # -----------------------
            # Test CSV
            # -----------------------

            if test_path and os.path.exists(
                    test_path
            ):

                with open(
                    test_path,
                    "rb"
                ) as f:

                    col2.download_button(

                        "Test CSV",

                        data=f.read(),

                        file_name=
                        "test.csv",

                        mime=
                        "text/csv"
                    )


            # -----------------------
            # Preprocessor.pkl
            # -----------------------

            if preprocessor_path and os.path.exists(
                    preprocessor_path
            ):

                with open(
                    preprocessor_path,
                    "rb"
                ) as f:

                    col3.download_button(

                        "Preprocessor",

                        data=f.read(),

                        file_name=
                        "preprocessor.pkl",

                        mime=
                        "application/octet-stream"
                    )


            # -----------------------
            # Processed Train
            # -----------------------

            if processed_train and os.path.exists(
                    processed_train
            ):

                with open(
                    processed_train,
                    "rb"
                ) as f:

                    col4.download_button(

                        "Processed Train",

                        data=f.read(),

                        file_name=
                        "processed_train.csv",

                        mime=
                        "text/csv"
                    )


            # -----------------------
            # Processed Test
            # -----------------------

            if processed_test and os.path.exists(
                    processed_test
            ):

                with open(
                    processed_test,
                    "rb"
                ) as f:

                    col5.download_button(

                        "Processed Test",

                        data=f.read(),

                        file_name=
                        "processed_test.csv",

                        mime=
                        "text/csv"
                    )
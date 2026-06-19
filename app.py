import os
import tempfile
import uuid
import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command
from workflow import graph
from utils.create_pipeline_bundle_zip import create_pipeline_bundle_zip

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataPrep Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS — design system
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Tokens ──────────────────────────────────────────────── */
:root {
    --bg:        #0D1117;
    --surface:   #161B22;
    --border:    #21262D;
    --accent:    #3FB950;        /* terminal-green — data is alive */
    --accent2:   #58A6FF;        /* cool blue for info */
    --warn:      #D29922;
    --error:     #F85149;
    --text:      #E6EDF3;
    --muted:     #8B949E;
    --radius:    10px;
    --mono:      'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    --sans:      'Inter', system-ui, sans-serif;
}

/* ── Reset / base ────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* Hide Streamlit branding */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ── Hero header ─────────────────────────────────────────── */
.dp-hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.dp-hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.dp-hero-title {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 0.4rem;
    color: var(--text);
}
.dp-hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
}

/* ── Upload zone ─────────────────────────────────────────── */
.dp-upload-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--muted) !important;
    font-size: 0.88rem !important;
}

/* ── Step log ────────────────────────────────────────────── */
.dp-steps-header {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.dp-steps-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.dp-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.65rem 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    margin-bottom: 0.45rem;
    font-size: 0.88rem;
    color: var(--text);
    animation: fadeSlide 0.25s ease both;
}
.dp-step-icon {
    font-size: 0.85rem;
    margin-top: 0.05rem;
    flex-shrink: 0;
    color: var(--accent);
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Cards ───────────────────────────────────────────────── */
.dp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.dp-card-title {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.dp-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
}
.dp-card-accent { color: var(--accent); }
.dp-card-blue   { color: var(--accent2); }

/* ── Column list ─────────────────────────────────────────── */
.dp-col-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.5rem;
}
.dp-col-tag {
    font-family: var(--mono);
    font-size: 0.73rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    background: rgba(63, 185, 80, 0.1);
    color: var(--accent);
    border: 1px solid rgba(63, 185, 80, 0.25);
}
.dp-col-tag.cat {
    background: rgba(88, 166, 255, 0.1);
    color: var(--accent2);
    border-color: rgba(88, 166, 255, 0.25);
}

/* ── Interrupt / approval box ────────────────────────────── */
.dp-interrupt {
    background: var(--surface);
    border: 1px solid var(--warn);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.dp-interrupt-eyebrow {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--warn);
    margin-bottom: 0.75rem;
}
.dp-interrupt-title {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.35rem;
    color: var(--text);
}
.dp-interrupt-body {
    font-size: 0.88rem;
    color: var(--muted);
    margin: 0 0 1.25rem;
}
.dp-kv-row {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
}
.dp-kv-key {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    min-width: 100px;
}
.dp-kv-val {
    color: var(--text);
    font-weight: 500;
}

/* ── Error / info banners ────────────────────────────────── */
.dp-banner {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    padding: 1rem 1.25rem;
    border-radius: var(--radius);
    margin-bottom: 0.85rem;
    font-size: 0.88rem;
}
.dp-banner.error {
    background: rgba(248, 81, 73, 0.08);
    border: 1px solid rgba(248, 81, 73, 0.35);
    color: #F85149;
}
.dp-banner.info {
    background: rgba(88, 166, 255, 0.08);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: var(--accent2);
}
.dp-banner.success {
    background: rgba(63, 185, 80, 0.08);
    border: 1px solid rgba(63, 185, 80, 0.3);
    color: var(--accent);
}
.dp-banner-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.dp-banner-msg  { line-height: 1.55; }

/* ── Loading pulse ───────────────────────────────────────── */
.dp-loading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.dp-spinner {
    width: 16px; height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Download button ─────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: var(--accent) !important;
    color: #0D1117 !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.85 !important; }

/* ── Primary / secondary buttons ────────────────────────── */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.77rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 1.1rem !important;
    transition: background 0.15s, color 0.15s !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--accent) !important;
    color: #0D1117 !important;
}

/* Skip / secondary variant (second column buttons) */
.dp-secondary [data-testid="stButton"] > button {
    border-color: var(--border) !important;
    color: var(--muted) !important;
}
.dp-secondary [data-testid="stButton"] > button:hover {
    background: var(--border) !important;
    color: var(--text) !important;
}

/* ── Select box ──────────────────────────────────────────── */
[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: 0.82rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
}

/* ── JSON viewer ─────────────────────────────────────────── */
[data-testid="stJson"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── Status text / spinner ───────────────────────────────── */
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Section divider ─────────────────────────────────────── */
.dp-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Metric override ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.7rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper rendering utilities
# ──────────────────────────────────────────────────────────────────────────────

def hero():
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown("""
        <div class="dp-hero">
            <div class="dp-hero-eyebrow">Automated ML</div>
            <h1 class="dp-hero-title">Data Preprocessing Agent</h1>
            <p class="dp-hero-sub">Upload a CSV — the agent cleans, splits, encodes, and packages your preprocessing pipeline automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        # Only show when a run exists
        if "result" in st.session_state:
            st.markdown("<div style='padding-top:2.5rem'>", unsafe_allow_html=True)
            if st.button("↺  New Run"):
                new_thread_id()
            st.markdown("</div>", unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="dp-steps-header">{text}</div>', unsafe_allow_html=True)


def step_item(msg: str):
    st.markdown(f"""
    <div class="dp-step">
        <span class="dp-step-icon">✓</span>
        <span>{msg}</span>
    </div>""", unsafe_allow_html=True)


def loading_row(msg: str):
    st.markdown(f"""
    <div class="dp-loading">
        <div class="dp-spinner"></div>
        <span>{msg}</span>
    </div>""", unsafe_allow_html=True)


def banner(kind: str, icon: str, msg: str):
    """kind = 'error' | 'info' | 'success'"""
    st.markdown(f"""
    <div class="dp-banner {kind}">
        <span class="dp-banner-icon">{icon}</span>
        <span class="dp-banner-msg">{msg}</span>
    </div>""", unsafe_allow_html=True)


def kv_row(key: str, val):
    st.markdown(f"""
    <div class="dp-kv-row">
        <span class="dp-kv-key">{key}</span>
        <span class="dp-kv-val">{val}</span>
    </div>""", unsafe_allow_html=True)


def col_tags(cols: list, variant: str = ""):
    cls = f"dp-col-tag {variant}".strip()
    tags = "".join(f'<span class="{cls}">{c}</span>' for c in cols)
    st.markdown(f'<div class="dp-col-list">{tags}</div>', unsafe_allow_html=True)

def new_thread_id():
    """Generate fresh thread, wipe old session state, redirect."""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # New thread ID in URL → triggers fresh run
    st.query_params["thread_id"] = str(uuid.uuid4())
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────────────────────────

# ── Thread ID — lives in URL so refresh restores the same run ────────────────
if "thread_id" not in st.query_params:
    st.query_params["thread_id"] = str(uuid.uuid4())

thread_id = st.query_params["thread_id"]
config = {"configurable": {"thread_id": thread_id}}

# ── Restore state from Supabase checkpoint on refresh ────────────────────────
if "result" not in st.session_state:
    try:
        saved = graph.get_state(config)
        if saved and saved.values.get("steps"):
            st.session_state["result"]   = dict(saved.values)
            st.session_state["temp_dir"] = saved.values.get("temp_dir", "")
            st.session_state["report_path"] = os.path.join(
                saved.values.get("temp_dir", ""), "ydata_report.html"
            )
    except Exception:
        pass  # no prior checkpoint — fresh run
# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

hero()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="dp-upload-label">Input Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a CSV file here, or click to browse",
    type=["csv"],
    label_visibility="collapsed",
)

# ── Pipeline run ──────────────────────────────────────────────────────────────
if uploaded_file is not None:

    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp()

    temp_dir  = st.session_state["temp_dir"]
    input_path  = os.path.join(temp_dir, "input.csv")
    output_path = os.path.join(temp_dir, "processed.csv")
    report_path = os.path.join(temp_dir, "ydata_report.html")
    st.session_state["report_path"] = report_path

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "result" not in st.session_state:

        section_label("Pipeline")
        status_placeholder = st.empty()
        steps_placeholder  = st.empty()

        collected_steps: list[str] = []
        latest = None
        pipeline_error = None

        try:
            for chunk in graph.stream(
                {
                    "input_file_path":  input_path,
                    "output_file_path": output_path,
                    "report_path":      report_path,
                    "temp_dir":         temp_dir,
                    "steps":            [],
                    "numerical_pipeline_path":  None,
                    "categorical_pipeline_path": None,
                    "final_preprocessor_path":  None,
                },
                config=config,
                stream_mode="values",
            ):
                latest = chunk
                steps = chunk.get("steps") or []

                if steps:
                    collected_steps = steps
                    with steps_placeholder.container():
                        for s in steps:
                            step_item(s)

                if "__interrupt__" in chunk:
                    with status_placeholder.container():
                        loading_row("Waiting for your input…")
                else:
                    current = steps[-1] if steps else "initializing"
                    with status_placeholder.container():
                        loading_row(current)

        except Exception as exc:
            pipeline_error = str(exc)

        if pipeline_error:
            status_placeholder.empty()
            banner("error", "✕", f"Pipeline error: {pipeline_error}")
        else:
            status_placeholder.empty()
            result = latest
            if result and "__interrupt__" not in result:
                banner("success", "✓", "All pipeline stages complete.")
            st.session_state["result"] = result

# ── Results & interactions ────────────────────────────────────────────────────
if "result" in st.session_state:

    result = st.session_state["result"]

    # Steps log (already ran but re-render on rerun)
    steps = result.get("steps", [])
    if steps:
        section_label("Completed Steps")
        for s in steps:
            step_item(s)

    st.markdown('<hr class="dp-divider">', unsafe_allow_html=True)

    # Dataset report
    report_path = st.session_state.get("report_path")
    if report_path and os.path.exists(report_path):
        section_label("Dataset Profile Report")
        with st.spinner("Rendering profiling report…"):
            with open(report_path, "r", encoding="utf-8") as html_file:
                components.html(html_file.read(), height=800, scrolling=True)
        st.markdown('<hr class="dp-divider">', unsafe_allow_html=True)

    # ── INTERRUPT handlers ────────────────────────────────────────────────────
    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value
        interrupt_type = interrupt_data.get("type")

        # ── Target column confirmation ────────────────────────────────────────
        if interrupt_type == "target_confirmation":

            detected_target = interrupt_data.get("detected_target", "")
            confidence      = interrupt_data.get("confidence", "")
            reason          = interrupt_data.get("reason", "")
            columns         = interrupt_data.get("columns", [])

            st.markdown("""
            <div class="dp-interrupt">
                <div class="dp-interrupt-eyebrow">⚠ Action required — Target column</div>
                <div class="dp-interrupt-title">Confirm the prediction target</div>
                <div class="dp-interrupt-body">
                    The agent detected a likely target column. Confirm it or pick a different one before the pipeline continues.
                </div>
            """, unsafe_allow_html=True)

            kv_row("Detected",   f"<code>{detected_target}</code>")
            kv_row("Confidence", confidence)
            kv_row("Reason",     reason)
            st.markdown("</div>", unsafe_allow_html=True)

            selected_target = st.selectbox(
                "Override target column (optional)",
                options=["No target column"] + columns,
                index=(columns.index(detected_target) + 1) if detected_target in columns else 0,
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓  Confirm detected target"):
                    try:
                        loading_row(f"Resuming with target: {detected_target}…")
                        result = graph.invoke(Command(resume={"approved": True}), config=config)
                        st.session_state["result"] = result
                        st.rerun()
                    except Exception as exc:
                        banner("error", "✕", f"Resume failed: {exc}")

            with col2:
                st.markdown('<div class="dp-secondary">', unsafe_allow_html=True)
                if st.button("↩  Use selected / unsupervised"):
                    resume_data = (
                        {"approved": False, "target_column": None}
                        if selected_target == "No target column"
                        else {"approved": False, "target_column": selected_target}
                    )
                    try:
                        loading_row("Resuming…")
                        result = graph.invoke(Command(resume=resume_data), config=config)
                        st.session_state["result"] = result
                        st.rerun()
                    except Exception as exc:
                        banner("error", "✕", f"Resume failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Feature engineering approval ──────────────────────────────────────
        elif interrupt_type == "feature_engineering_plan":

            plan = interrupt_data.get("plan", {})

            st.markdown("""
            <div class="dp-interrupt">
                <div class="dp-interrupt-eyebrow">⚠ Action required — Feature engineering</div>
                <div class="dp-interrupt-title">Review the proposed transformations</div>
                <div class="dp-interrupt-body">
                    Approve to apply these features, or skip to use raw columns only.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.json(plan)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓  Approve feature plan"):
                    try:
                        loading_row("Applying feature engineering…")
                        result = graph.invoke(Command(resume={"approved": True}), config=config)
                        st.session_state["result"] = result
                        st.rerun()
                    except Exception as exc:
                        banner("error", "✕", f"Resume failed: {exc}")

            with col2:
                st.markdown('<div class="dp-secondary">', unsafe_allow_html=True)
                if st.button("↩  Skip feature engineering"):
                    try:
                        result = graph.invoke(Command(resume={"approved": False}), config=config)
                        st.session_state["result"] = result
                        st.rerun()
                    except Exception as exc:
                        banner("error", "✕", f"Resume failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Final results ─────────────────────────────────────────────────────────
    else:

        # Errors
        for err_key, label in [
            ("numerical_error",   "Numerical pipeline error"),
            ("categorical_error", "Categorical pipeline error"),
            ("error",             "Pipeline error"),
        ]:
            if result.get(err_key):
                banner("error", "✕", f"{label}: {result[err_key]}")

        if not result.get("error"):

            # Success message
            if result.get("message"):
                banner("success", "✓", result["message"])

            # Problem type + target
            meta_pairs = [
                ("Problem type",   result.get("problem_type")),
                ("Target column",  result.get("target_column")),
            ]
            shown = [(k, v) for k, v in meta_pairs if v]
            if shown:
                section_label("Run Summary")
                for k, v in shown:
                    kv_row(k, v)

            st.markdown('<hr class="dp-divider">', unsafe_allow_html=True)

            # Train / test split
            train_rows = result.get("train_rows")
            test_rows  = result.get("test_rows")
            if train_rows and test_rows:
                section_label("Train / Test Split")
                col_a, col_b = st.columns(2)
                col_a.metric("Training rows",  train_rows)
                col_b.metric("Testing rows",   test_rows)
                st.markdown('<hr class="dp-divider">', unsafe_allow_html=True)

            # Column classification
            num_cols = result.get("numerical_columns", [])
            cat_cols = result.get("categorical_columns", [])
            if num_cols or cat_cols:
                section_label("Column Classification")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown('<div class="dp-card-title">Numerical</div>', unsafe_allow_html=True)
                    col_tags(num_cols, variant="")
                with col_d:
                    st.markdown('<div class="dp-card-title">Categorical</div>', unsafe_allow_html=True)
                    col_tags(cat_cols, variant="cat")
                st.markdown('<hr class="dp-divider">', unsafe_allow_html=True)

            # Downloads
            preprocessor_path   = result.get("final_preprocessor_path")
            train_path          = result.get("train_path")
            test_path           = result.get("test_path")
            processed_train     = result.get("processed_train_path")
            processed_test      = result.get("processed_test_path")
            feature_plan_path   = result.get("feature_engineering_plan_path")
            feat_transformer    = result.get("feature_engineering_transformer_path")

            bundle_zip_path = create_pipeline_bundle_zip(
                temp_dir=st.session_state["temp_dir"],
                preprocessor_path=preprocessor_path,
                train_path=train_path,
                test_path=test_path,
                processed_train_path=processed_train,
                processed_test_path=processed_test,
                feature_plan_path=feature_plan_path,
                feature_engineering_transformer_path=feat_transformer,
            )

            if bundle_zip_path and os.path.exists(bundle_zip_path):
                section_label("Export")
                st.markdown(
                    '<p style="color:var(--muted);font-size:0.85rem;margin-bottom:0.75rem;">'
                    "Contains the fitted preprocessor, train/test splits, processed arrays, "
                    "and (if applicable) the feature engineering transformer."
                    "</p>",
                    unsafe_allow_html=True,
                )
                with open(bundle_zip_path, "rb") as f:
                    st.download_button(
                        "↓  Download Pipeline Bundle  (.zip)",
                        data=f.read(),
                        file_name="pipeline_bundle.zip",
                        mime="application/zip",
                    )
            
            else:
                if not result.get("error"):
                    banner("info", "ℹ", "No downloadable bundle was generated for this run.")
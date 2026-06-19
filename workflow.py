
import os
from importlib import import_module
from typing import Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover - import path varies by langgraph version
    MemorySaver = None

PostgresSaver = None
ConnectionPool = None

from nodes.load_dataset_node import load_dataset_node
from nodes.cleanup import cleanup
from nodes.target_detection import target_detection_node
from nodes.split_and_classify import split_and_classify_node
from nodes.feature_engineering_node import feature_engineering_node
from nodes.classify_columns_node import classify_columns_node
from nodes.categorical_preprocessing_node import categorical_preprocessing_node
from nodes.numerical_preprocessing_node import numerical_preprocessing_node
from nodes.merge_preprocessors_node import merge_preprocessors_node

def build_checkpointer():
    db_uri = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")

    global PostgresSaver, ConnectionPool

    if db_uri and PostgresSaver is None and ConnectionPool is None:
        try:
            postgres_module_name = "langgraph." + "checkpoint.postgres"
            psycopg_pool_module_name = "psycopg_" + "pool"
            postgres_module = import_module(postgres_module_name)
            psycopg_pool_module = import_module(psycopg_pool_module_name)
            PostgresSaver = getattr(postgres_module, "PostgresSaver", None)
            ConnectionPool = getattr(psycopg_pool_module, "ConnectionPool", None)
        except Exception:
            PostgresSaver = None
            ConnectionPool = None

    if db_uri and PostgresSaver and ConnectionPool:
        pool = ConnectionPool(
            conninfo=db_uri,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return checkpointer

    if MemorySaver is not None:
        return MemorySaver()

    return None


class MLState(TypedDict):
    input_file_path: str
    output_file_path: str
    report_path: str
    temp_dir: Optional[str]

    rows: Optional[int]
    cols: Optional[int]

    target_column: Optional[str]
    problem_type: Optional[str]

    # Split paths
    train_path: Optional[str]
    test_path: Optional[str]
    train_raw_path: Optional[str]
    test_raw_path: Optional[str]
    train_rows: Optional[int]
    test_rows: Optional[int]

    # Column classification from LLM
    numerical_columns: Optional[list[str]]
    categorical_columns: Optional[list[str]]

    numerical_pipeline_path: Optional[str]
    categorical_pipeline_path: Optional[str]
    final_preprocessor_path: Optional[str]
    apply_smote: Optional[bool]
    smote_reason: Optional[str]
    smote_applied: Optional[bool]

    processed_train_path: Optional[str]
    processed_test_path: Optional[str]

    feature_engineering_plan: Optional[dict]
    feature_engineering_plan_path: Optional[str]
    feature_engineering_approved: Optional[bool]
    feature_engineering_applied: Optional[bool]
    feature_engineering_transformer_path:Optional[str]

    numerical_error: Optional[str]
    categorical_error: Optional[str]

    steps: Optional[list[str]]
    message: Optional[str]
    error: Optional[str]


builder = StateGraph(MLState)
def barrier_node(state):
    """Ensures split_and_classify state is fully committed before fan-out."""
    return {}

def error_router(state):
    return "end" if state.get("error") else "next"

builder.add_node("load_dataset", load_dataset_node)
builder.add_node("cleanup", cleanup)
builder.add_node("target_detection", target_detection_node)
builder.add_node("split_and_classify", split_and_classify_node)
builder.add_node("feature_engineering", feature_engineering_node)
builder.add_node("classify_columns", classify_columns_node)
builder.add_node("barrier", barrier_node)
builder.add_node("numerical_processing",numerical_preprocessing_node)
builder.add_node("categorical_processing",categorical_preprocessing_node)
builder.add_node("merge_preprocessors",merge_preprocessors_node)

builder.add_edge(START, "load_dataset")
builder.add_conditional_edges("load_dataset", error_router, {"next": "cleanup", "end": END})
builder.add_conditional_edges("cleanup", error_router, {"next": "target_detection", "end": END})
builder.add_conditional_edges("target_detection", error_router, {"next": "split_and_classify", "end": END})
builder.add_conditional_edges("split_and_classify", error_router, {"next": "feature_engineering", "end": END})
builder.add_conditional_edges("feature_engineering", error_router, {"next": "classify_columns", "end": END})
builder.add_conditional_edges("classify_columns", error_router, {"next": "barrier", "end": END})
builder.add_edge("barrier", "numerical_processing") # ← fan-out from barrier
builder.add_edge("barrier", "categorical_processing")
builder.add_edge("numerical_processing", "merge_preprocessors")
builder.add_edge("categorical_processing", "merge_preprocessors")
builder.add_edge("merge_preprocessors", END)


checkpointer = build_checkpointer()
graph = builder.compile(checkpointer=checkpointer) if checkpointer is not None else builder.compile()
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.load_dataset_node import load_dataset_node
from nodes.cleanup import cleanup
from nodes.target_detection import target_detection_node
from nodes.split_and_classify import split_and_classify_node
from nodes.categorical_preprocessing_node import categorical_preprocessing_node
from nodes.numerical_preprocessing_node import numerical_preprocessing_node
from nodes.merge_preprocessors_node import merge_preprocessors_node



class MLState(TypedDict):
    input_file_path: str
    output_file_path: str
    report_path: str

    rows: Optional[int]
    cols: Optional[int]

    target_column: Optional[str]
    problem_type: Optional[str]

    # Split paths
    train_path: Optional[str]
    test_path: Optional[str]
    train_rows: Optional[int]
    test_rows: Optional[int]

    # Column classification from LLM
    numerical_columns: Optional[list[str]]
    categorical_columns: Optional[list[str]]

    numerical_pipeline_path: Optional[str]

    
    categorical_pipeline_path: Optional[str]


    final_preprocessor_path: Optional[str]

    steps: Optional[list[str]]

    message: Optional[str]
    error: Optional[str]

    steps: Optional[list[str]]
    processed_train_path:Optional[str]

    processed_test_path:Optional[str]

    final_preprocessor_path:Optional[str]

    message: Optional[str]
    error: Optional[str]


builder = StateGraph(MLState)
def barrier_node(state):
    """Ensures split_and_classify state is fully committed before fan-out."""
    return {}

builder.add_node("load_dataset", load_dataset_node)
builder.add_node("cleanup", cleanup)
builder.add_node("target_detection", target_detection_node)
builder.add_node("split_and_classify", split_and_classify_node)
builder.add_node("barrier", barrier_node)
builder.add_node("numerical_processing",numerical_preprocessing_node)
builder.add_node("categorical_processing",categorical_preprocessing_node)
builder.add_node("merge_preprocessors",merge_preprocessors_node)

builder.add_edge(START, "load_dataset")
builder.add_edge("load_dataset", "cleanup")
builder.add_edge("cleanup", "target_detection")
builder.add_edge("target_detection", "split_and_classify")
builder.add_edge("split_and_classify", "barrier")   # ← go through barrier first
builder.add_edge("barrier", "numerical_processing") # ← fan-out from barrier
builder.add_edge("barrier", "categorical_processing")
builder.add_edge("numerical_processing", "merge_preprocessors")
builder.add_edge("categorical_processing", "merge_preprocessors")
builder.add_edge("merge_preprocessors", END)




checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
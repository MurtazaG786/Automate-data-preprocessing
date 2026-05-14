from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.load_dataset_node import load_dataset_node
from nodes.cleanup import cleanup
from nodes.target_detection import target_detection_node
from nodes.split_and_classify import split_and_classify_node


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

    steps: Optional[list[str]]

    message: Optional[str]
    error: Optional[str]


builder = StateGraph(MLState)

builder.add_node("load_dataset", load_dataset_node)
builder.add_node("cleanup", cleanup)
builder.add_node("target_detection", target_detection_node)
builder.add_node("split_and_classify", split_and_classify_node)

builder.add_edge(START, "load_dataset")
builder.add_edge("load_dataset", "cleanup")
builder.add_edge("cleanup", "target_detection")
builder.add_edge("target_detection", "split_and_classify")
builder.add_edge("split_and_classify", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
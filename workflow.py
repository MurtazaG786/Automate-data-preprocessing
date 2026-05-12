from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.load_dataset_node import load_dataset_node
from nodes.cleanup import cleanup
from nodes.target_detection import target_detection_node


class MLState(TypedDict):
    input_file_path: str
    output_file_path: str
    report_path: str

    rows: Optional[int]
    cols: Optional[int]

    target_column: Optional[str]
    problem_type: Optional[str]

    steps: Optional[list[str]]

    message: Optional[str]
    error: Optional[str]


builder = StateGraph(MLState)

builder.add_node("load_dataset", load_dataset_node)
builder.add_node("cleanup", cleanup)
builder.add_node("target_detection", target_detection_node)

builder.add_edge(START, "load_dataset")
builder.add_edge("load_dataset", "cleanup")
builder.add_edge("cleanup", "target_detection")
builder.add_edge("target_detection", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
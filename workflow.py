from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import pandas as pd
from nodes.load_dataset_node import load_dataset_node
from nodes.cleanup import column_cleanup

class MLState(TypedDict):
    uploaded_file: object
    df: Optional[pd.DataFrame]
    rows: Optional[int]
    cols: Optional[int]
    message: Optional[str]
    error: Optional[str]

builder = StateGraph(MLState)

builder.add_node("load_dataset", load_dataset_node)
builder.add_node("cleanup",column_cleanup)

builder.add_edge(START, "load_dataset")
builder.add_edge("load_dataset","cleanup")
builder.add_edge("cleanup", END)

graph = builder.compile()

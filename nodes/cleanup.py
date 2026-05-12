import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()


class ColumnsToDrop(BaseModel):
    columns: list[str] = Field(
        description="List of unwanted column names to drop from dataframe"
    )


def cleanup(state):
    input_path = state.get("input_file_path")
    output_path = state.get("output_file_path")

    if not input_path or not os.path.exists(input_path):
        return {"error": "No input file path found in state."}

    if not output_path:
        return {"error": "No output file path found in state."}

    df = pd.read_csv(input_path, encoding="latin1")

    original_len = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = original_len - len(df)

    valid_columns_to_drop = []

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if api_key and model_name:
        prompt = f"""
    You are a senior machine learning engineer. Analyze the dataset and identify unwanted columns. Unwanted columns may include: - ID columns - serial number columns - constant columns - columns with mostly missing values - irrelevant text columns Return only valid column names from the given dataframe.

    Column names and dtypes:
    {df.dtypes.to_string()}

    Dataset sample:
    {df.head(10).to_string()}
    """

        client = genai.Client(api_key=api_key)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ColumnsToDrop,
                ),
            )

            result = ColumnsToDrop.model_validate_json(response.text)

            valid_columns_to_drop = [
                col for col in result.columns
                if col in df.columns
            ]

            if valid_columns_to_drop:
                df.drop(columns=valid_columns_to_drop, inplace=True)

        except Exception:
            valid_columns_to_drop = []

    df.to_csv(output_path, index=False)

    return {
        "output_file_path": output_path,
        "steps": state.get("steps", []) + [
            f"Cleanup completed. Duplicates removed: {duplicates_removed}. Columns dropped: {valid_columns_to_drop}"
        ],
        "message": f"Cleanup completed. Columns dropped: {json.dumps(valid_columns_to_drop)}",
        "error": None
    }
import json
import os
import re

import pandas as pd
import google.genai as genai
from dotenv import load_dotenv
load_dotenv()

def column_cleanup(state):
    df_path = state.get("df_path")
    if not df_path or not os.path.exists(df_path):
        return {"error": "No dataframe file path in state."}

    df = pd.read_csv(df_path, encoding="latin1")

    duplicate_count = df.duplicated().sum()
    if duplicate_count != 0:
        df.drop_duplicates(inplace=True)
    
    prompt=f""" hey you're a senior machine learning engineer who have worked on so many datasets ,
    analyze this dataframe and return me the list of unwanted column names same as given in the dataframe ,
    just return the list of unwanted column names in json format nothing else
    here is your dataset :
    {df.head(50)}
      """
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    if not api_key or not model_name:
        return {"error": "Missing GOOGLE_API_KEY or MODEL_NAME in environment."}

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
    except Exception as exc:
        return {"error": f"LLM request failed: {exc}"}

    column_info = (response.text or "").strip()
    column_info = re.sub(r"```json|```", "", column_info).strip()
    try:
        columns_to_drop = json.loads(column_info)
    except json.JSONDecodeError:
        return {"error": "Model output is not valid JSON list of column names."}

    df.drop(columns=columns_to_drop, inplace=True)

    df.to_csv(df_path, index=False)
    message = json.dumps(columns_to_drop)

    return {
        "df_path": df_path,
        "message": f"columns dropped succesfully {message}",
        "error": None
    }

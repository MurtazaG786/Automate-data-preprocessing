import pandas as pd


def load_dataset_node(state):

    try:
        file = state["uploaded_file"]

        # FILE CHECK
        if file is None:
            return {
                "error": "No file uploaded."
            }

        # CSV CHECK
        if not file.name.endswith(".csv"):
            return {
                "error": "Only CSV files are supported."
            }

        # LOAD DATASET
        df = pd.read_csv(file)

        # EMPTY CHECK
        if df.empty:
            return {
                "error": "Uploaded CSV file is empty."
            }

        # RETURN STATE UPDATE
        return {
            "df": df,

            "rows": df.shape[0],
            "cols": df.shape[1],

            "message": "Dataset loaded successfully.",

            "error": None
        }

    except Exception as e:

        return {
            "error": str(e)
        }
import pandas as pd
from ydata_profiling import ProfileReport
import sweetviz as sv



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

        #profiling
        profile = ProfileReport(df, title="Dataset Report")
        profile.to_file("ydata_report.html")

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
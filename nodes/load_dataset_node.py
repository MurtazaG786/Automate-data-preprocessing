import pandas as pd

try:
    from ydata_profiling import ProfileReport
except Exception:  # pragma: no cover - optional deployment dependency
    ProfileReport = None


def load_dataset_node(state):

    try:
        input_path = state["input_file_path"]
        report_path = state["report_path"]

        # file exists check
        if not input_path:
            return {
                "error": "No input file path provided."
            }

        # csv check
        if not input_path.endswith(".csv"):
            return {
                "error": "Only CSV files are supported."
            }

        # read csv
        df = pd.read_csv(input_path)

        # empty file check
        if df.empty:
            return {
                "error": "Uploaded CSV file is empty."
            }

        # generate report when the profiling dependency is available
        if ProfileReport is not None:
            try:
                profile = ProfileReport(
                    df,
                    title="Dataset Report"
                )
                profile.to_file(report_path)
            except Exception:
                report_path = None

        return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "steps": state.get("steps", []) + [
            f"Dataset loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        ],
        "message": "Dataset loaded successfully.",
        "error": None
    }
            
    except Exception as e:
        return {
            "error": str(e)
        }
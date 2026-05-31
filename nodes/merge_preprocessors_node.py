import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import tempfile

def merge_preprocessors_node(state):

    if state.get("numerical_error"):
        return {"error": state.get("numerical_error")}

    if state.get("categorical_error"):
        return {"error": state.get("categorical_error")}

    train = pd.read_csv(
        state["train_path"]
    )

    test = pd.read_csv(
        state["test_path"]
    )

    target = state.get(
        "target_column"
    )


    # -------------------------
    # Separate X and y
    # -------------------------

    if target:

        X_train = train.drop(
            columns=[target]
        )

        X_test = test.drop(
            columns=[target]
        )

        y_train = train[target]

        y_test = test[target]

    else:

        X_train = train
        X_test = test

        y_train = None
        y_test = None


    num_cols = state.get("numerical_columns") or []
    cat_cols = state.get("categorical_columns") or []
    num_path = state.get("numerical_pipeline_path")
    cat_path = state.get("categorical_pipeline_path")

    transformers = []

    if num_cols:
        if not num_path or not os.path.exists(num_path):
            return {"error": "Numerical preprocessing pipeline not found."}
        numerical_pipeline = joblib.load(num_path)
        transformers.append(("num", numerical_pipeline, num_cols))

    if cat_cols:
        if not cat_path or not os.path.exists(cat_path):
            return {"error": "Categorical preprocessing pipeline not found."}
        categorical_pipeline = joblib.load(cat_path)
        transformers.append(("cat", categorical_pipeline, cat_cols))


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )
    print(preprocessor)

   

    # -------------------------
    # Fit transform X
    # -------------------------

    train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out()
    )

    test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out()
    )


    # -------------------------
    # Encode target if categorical
    # -------------------------

    if target:

        if (
            y_train.dtype == "object"
            or y_train.nunique() < 20
        ):

            encoder = LabelEncoder()

            y_train = encoder.fit_transform(
                y_train.astype(str)
            )

            y_test = encoder.transform(
                y_test.astype(str)
            )


        train_processed[
            target
        ] = y_train


        test_processed[
            target
        ] = y_test


  

    # temporary csv files
    train_temp = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    )

    test_temp = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    )

    train_processed.to_csv(
        train_temp.name,
        index=False
    )

    test_processed.to_csv(
        test_temp.name,
        index=False
    )


    # temporary preprocessor file
    preprocessor_temp = tempfile.NamedTemporaryFile(
        suffix=".pkl",
        delete=False
    )

    joblib.dump(
        preprocessor,
        preprocessor_temp.name
    )


    train_out = train_temp.name
    test_out = test_temp.name
    preprocessor_path = preprocessor_temp.name


    return {

        "processed_train_path":
            train_out,

        "processed_test_path":
            test_out,

        "final_preprocessor_path":
            preprocessor_path,

        "steps": state.get("steps", []) + [
            "Numerical preprocessing ready.",
            "Categorical preprocessing ready.",
            "Merged preprocessors.",
            "Processed train/test saved.",
        ]
    }
import os, zipfile
def create_pipeline_bundle_zip(
    temp_dir: str,
    preprocessor_path: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    processed_train_path: str | None = None,
    processed_test_path: str | None = None,
    feature_plan_path: str | None = None,
    feature_engineering_transformer_path: str | None = None,
) -> str:
    zip_path = os.path.join(temp_dir, "pipeline_bundle.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        # -----------------------------
        # Custom transformer module
        # -----------------------------
        if feature_engineering_transformer_path and os.path.exists(feature_engineering_transformer_path):
            zipf.write(
                feature_engineering_transformer_path,
                arcname="nodes/feature_engineering_transformer.py"
            )

            zipf.writestr("nodes/__init__.py", "")

        # -----------------------------
        # Preprocessor pickle
        # -----------------------------
        if preprocessor_path and os.path.exists(preprocessor_path):
            zipf.write(
                preprocessor_path,
                arcname="preprocessor.pkl"
            )

        # -----------------------------
        # Feature engineering plan
        # -----------------------------
        if feature_plan_path and os.path.exists(feature_plan_path):
            zipf.write(
                feature_plan_path,
                arcname="feature_plan.json"
            )

        # -----------------------------
        # Train/Test CSV files
        # -----------------------------
        if train_path and os.path.exists(train_path):
            zipf.write(
                train_path,
                arcname="train.csv"
            )

        if test_path and os.path.exists(test_path):
            zipf.write(
                test_path,
                arcname="test.csv"
            )

        # -----------------------------
        # Processed train/test files
        # -----------------------------
        if processed_train_path and os.path.exists(processed_train_path):
            zipf.write(
                processed_train_path,
                arcname="processed_train.csv"
            )

        if processed_test_path and os.path.exists(processed_test_path):
            zipf.write(
                processed_test_path,
                arcname="processed_test.csv"
            )

        # -----------------------------
        # Requirements
        # -----------------------------
        requirements_txt = """pandas
numpy
scikit-learn
joblib
"""

        zipf.writestr("requirements.txt", requirements_txt)

        # -----------------------------
        # User guide
        # -----------------------------
        readme_txt = """# Pipeline Bundle

This bundle contains the generated preprocessing pipeline.

Files:
- preprocessor.pkl
- feature_plan.json
- nodes/feature_engineering_transformer.py
- nodes/__init__.py
- train.csv
- test.csv
- processed_train.csv
- processed_test.csv
- requirements.txt

Important:
Keep the nodes folder in the same directory as preprocessor.pkl.

Example usage:

import joblib
import pandas as pd

from nodes.feature_engineering_transformer import FeatureEngineeringTransformer

preprocessor = joblib.load("preprocessor.pkl")

df = pd.read_csv("train.csv")
processed = preprocessor.transform(df)

print(processed)
"""

        zipf.writestr("README.txt", readme_txt)

    return zip_path
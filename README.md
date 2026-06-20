# Automate Data Preprocessing

## Description
An automated ML Data Preprocessing Agent designed to streamline the machine learning data preparation phase. Built with Streamlit and LangGraph, this application loads your CSV datasets, profiles the data, detects the target column, splits and classifies features, and generates tailored preprocessing pipelines. It empowers data scientists and developers by handling the heavy lifting of data cleaning and feature engineering, exporting a ready-to-use, reusable bundle containing the fitted preprocessor and processed data.

## Features
- **Automated Data Profiling**: Generates comprehensive dataset reports using `ydata-profiling`.
- **Intelligent Target Detection**: Automatically identifies the most likely prediction target column, with an option to override or proceed without a target.
- **Smart Feature Classification**: Classifies columns into numerical and categorical features automatically.
- **Automated Feature Engineering**: Proposes and applies intelligent feature engineering transformations, with an interactive approval step.
- **Pipeline Generation**: Constructs robust `scikit-learn` numerical and categorical preprocessing pipelines.
- **Exportable Bundles**: Downloads a packaged ZIP bundle containing the fitted preprocessor, train/test splits, processed data, and feature engineering assets.
- **Stateful Checkpoints**: Persistent workflow states powered by LangGraph (in-memory or Postgres-backed).

## Installation
```bash
git clone https://github.com/MurtazaG786/Automate-data-preprocessing.git
cd Automate-data-preprocessing
python -m venv .venv
# On Windows use: .venv\Scripts\activate
# On macOS/Linux use: source .venv/bin/activate
pip install -r requirements.txt
```

Set up your environment variables by copying `.env.example` to `.env` (or create one):
```env
GOOGLE_API_KEY=your-api-key
MODEL_NAME=your-model-name
# Optional: For persistent LangGraph checkpoints
# SUPABASE_DB_URL=your-database-url
```

## Usage
Run the Streamlit application locally:
```bash
streamlit run app.py
```
or check-out the url :https://automate-data-preprocessing-os8of6jhfkgjjqyzt2x8nj.streamlit.app/

1. Open the provided local URL in your browser.
2. Drag and drop a CSV dataset into the upload zone.
3. The agent will begin analyzing the data. Review the Dataset Profile Report generated.
4. Interact with the application when prompted (e.g., confirming the target column, approving the feature engineering plan).
5. Once the pipeline finishes, click the **Download Pipeline Bundle (.zip)** button to get your processed data and `scikit-learn` preprocessors.

## Technologies Used
* **Python**: Core programming language.
* **Streamlit**: Web framework for the interactive user interface.
* **LangGraph**: Orchestration framework for the agentic workflow and stateful checkpoints.
* **LangChain & Google GenAI**: Used for intelligent decision making (like target detection and feature engineering plans).
* **Pandas & Scikit-learn**: Data manipulation and standard ML preprocessing pipelines.
* **ydata-profiling**: Exploratory data analysis and profiling reports.

## Contributors
* Aditikanojiya26
* MurtazaG786

## License
MIT License

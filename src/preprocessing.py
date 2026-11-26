
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Dataset configuration
DATASET_PATH = "dataset/loan_dataset.csv"

TARGET_COLUMN = "Loan_Status"

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


def load_dataset(path=DATASET_PATH):
    """Load and validate the loan prediction dataset."""

    try:
        df = pd.read_csv(path)

        required_columns = (
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
            + [TARGET_COLUMN]
        )

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # Remove records without a valid target
        df = df[df[TARGET_COLUMN].isin(["Y", "N"])].copy()

        # Convert target into numerical values
        y = df[TARGET_COLUMN].map({
            "Y": 1,
            "N": 0
        })

        # Remove target and ID from input features
        X = df.drop(
            columns=[TARGET_COLUMN, "Loan_ID"],
            errors="ignore"
        )

        return X, y

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )


def build_preprocessor():
    """Create preprocessing pipelines."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor

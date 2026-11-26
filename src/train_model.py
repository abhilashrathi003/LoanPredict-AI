
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from src.preprocessing import load_dataset, build_preprocessor


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model_evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# DEFINE MACHINE LEARNING MODELS
# ==========================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }


# ==========================================
# TRAIN AND EVALUATE MODELS
# ==========================================

def train_and_evaluate():

    print("\n" + "=" * 60)
    print("LOAN APPROVAL PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Load dataset
    X, y = load_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDataset Information")
    print("-" * 40)

    print("Total Records:", len(X))
    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))

    print("\nTraining Models...\n")

    models = get_models()

    results = []
    trained_models = {}

    # Train every algorithm
    for model_name, algorithm in models.items():

        print(f"Training {model_name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", algorithm)
        ])

        pipeline.fit(X_train, y_train)

        # Generate predictions
        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        macro_f1 = f1_score(
            y_test, predictions, average="macro",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test, probabilities
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Macro F1": macro_f1,
            "ROC AUC": roc_auc
        })

        trained_models[model_name] = pipeline

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"ROC-AUC: {roc_auc:.4f}"
        )

    # ======================================
    # MODEL COMPARISON
    # ======================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["Macro F1", "ROC AUC"],
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            formatters={
                col: "{:.4f}".format
                for col in results_df.select_dtypes(
                    include="number"
                ).columns
            }
        )
    )

    # Save comparison
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    # ======================================
    # SELECT BEST MODEL
    # ======================================

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("BEST MODEL:", best_model_name)
    print("=" * 60)

    print("\nDetailed Classification Report:\n")

    print(
        classification_report(
            y_test,
            best_predictions,
            labels=[0, 1],
            target_names=["Rejected", "Approved"],
            zero_division=0
        )
    )

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"]
    )

    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ======================================
    # ROC CURVE
    # ======================================

    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        best_probabilities,
        ax=ax,
        name=best_model_name
    )

    ax.set_title(f"ROC Curve - {best_model_name}")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    model_path = MODEL_DIR / "best_loan_model.joblib"

    joblib.dump(best_model, model_path)

    print("\nBest model saved successfully!")
    print("Model path:", model_path)

    print("\nEvaluation reports saved in:", REPORT_DIR)

    print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")


# ==========================================
# MAIN FUNCTION
# ==========================================

if __name__ == "__main__":

    try:
        train_and_evaluate()

    except Exception as error:
        print("\nERROR DURING MODEL TRAINING:")
        print(error)
        raise

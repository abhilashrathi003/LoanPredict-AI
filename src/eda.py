
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Project paths
DATASET_PATH = "dataset/loan_dataset.csv"
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    """Load dataset for exploratory analysis."""

    try:
        df = pd.read_csv(DATASET_PATH)

        if "Loan_Status" not in df.columns:
            raise ValueError("Loan_Status column not found.")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}"
        )


def save_chart(filename):
    """Save chart inside reports/eda."""

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Chart saved: {filename}")


def loan_status_distribution(df):
    """Visualize approved and rejected applications."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Loan_Status",
        hue="Loan_Status",
        palette=["#ef4444", "#22c55e"],
        legend=False
    )

    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Applications")

    save_chart("loan_status_distribution.png")


def applicant_income_distribution(df):
    """Visualize applicant income."""

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="ApplicantIncome",
        bins=30,
        kde=True,
        color="#2563eb"
    )

    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")

    save_chart("applicant_income_distribution.png")


def loan_amount_distribution(df):
    """Visualize loan amount distribution."""

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="LoanAmount",
        bins=30,
        kde=True,
        color="#8b5cf6"
    )

    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount (Thousands)")
    plt.ylabel("Frequency")

    save_chart("loan_amount_distribution.png")


def credit_history_analysis(df):
    """Analyze loan status against credit history."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Credit_History",
        hue="Loan_Status",
        palette={
            "Y": "#22c55e",
            "N": "#ef4444"
        }
    )

    plt.title("Credit History vs Loan Status")
    plt.xlabel("Credit History")
    plt.ylabel("Number of Applications")

    save_chart("credit_history_analysis.png")


def property_area_analysis(df):
    """Analyze loan approval by property area."""

    approval_rate = (
        df.groupby("Property_Area", observed=True)["Loan_Status"]
        .apply(lambda x: (x == "Y").mean() * 100)
        .reset_index(name="Approval Rate")
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=approval_rate,
        x="Property_Area",
        y="Approval Rate",
        hue="Property_Area",
        palette="viridis",
        legend=False
    )

    plt.title("Loan Approval Rate by Property Area")
    plt.xlabel("Property Area")
    plt.ylabel("Approval Rate (%)")
    plt.ylim(0, 100)

    save_chart("property_area_analysis.png")


def correlation_heatmap(df):
    """Display correlation between numerical features."""

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Numerical Feature Correlation Heatmap")

    save_chart("correlation_heatmap.png")


def missing_values_analysis(df):
    """Visualize missing values in the dataset."""

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:
        print("No missing values found.")
        return

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=missing.index,
        y=missing.values,
        hue=missing.index,
        palette="magma",
        legend=False
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Dataset Columns")
    plt.ylabel("Missing Records")
    plt.xticks(rotation=35, ha="right")

    save_chart("missing_values_analysis.png")


def main():
    print("=" * 50)
    print("LOAN APPROVAL - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    df = load_data()

    print("\nDataset Shape:", df.shape)

    print("\nDataset Information:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nLoan Status Counts:")
    print(df["Loan_Status"].value_counts())

    print("\nGenerating Visualizations...\n")

    loan_status_distribution(df)
    applicant_income_distribution(df)
    loan_amount_distribution(df)
    credit_history_analysis(df)
    property_area_analysis(df)
    correlation_heatmap(df)
    missing_values_analysis(df)

    print("\nEDA completed successfully!")
    print("Charts stored in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

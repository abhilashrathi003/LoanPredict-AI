
import sqlite3
from pathlib import Path
from datetime import datetime


# Database configuration
DATABASE_PATH = Path("database/loan_predictions.db")


def get_connection():
    """Create a connection to the SQLite database."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create the prediction history table."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                gender TEXT NOT NULL,
                married TEXT NOT NULL,
                dependents TEXT NOT NULL,
                education TEXT NOT NULL,
                self_employed TEXT NOT NULL,

                applicant_income REAL NOT NULL,
                coapplicant_income REAL NOT NULL,
                loan_amount REAL NOT NULL,
                loan_term INTEGER NOT NULL,

                credit_history REAL NOT NULL,
                property_area TEXT NOT NULL,

                prediction TEXT NOT NULL,

                approval_score REAL NOT NULL,
                rejection_score REAL NOT NULL,

                created_at TEXT NOT NULL

            )
        """)

        connection.commit()

        print("Database initialized successfully!")

    finally:
        connection.close()


def save_prediction(data, prediction,
                    approval_score, rejection_score):
    """Save a new prediction."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO predictions (

                gender,
                married,
                dependents,
                education,
                self_employed,

                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,

                credit_history,
                property_area,

                prediction,
                approval_score,
                rejection_score,
                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            data["Gender"],
            data["Married"],
            data["Dependents"],
            data["Education"],
            data["Self_Employed"],

            float(data["ApplicantIncome"]),
            float(data["CoapplicantIncome"]),
            float(data["LoanAmount"]),
            int(data["Loan_Amount_Term"]),

            float(data["Credit_History"]),
            data["Property_Area"],

            prediction,

            float(approval_score),
            float(rejection_score),

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        ))

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()


def get_all_predictions():
    """Retrieve all saved predictions."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """)

        records = cursor.fetchall()

        return [dict(record) for record in records]

    finally:
        connection.close()


def delete_prediction(prediction_id):
    """Delete a prediction using its ID."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM predictions WHERE id = ?",
            (int(prediction_id),)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()

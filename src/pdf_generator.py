from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def generate_prediction_pdf(record):
    """Generate a professional PDF report from a saved SQLite prediction record."""

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=29,
        textColor=colors.HexColor("#174EA6"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#174EA6"),
        spaceBefore=13,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
    )

    def create_table(data):
        """Create a consistently styled two-column report table."""
        table = Table(data, colWidths=[65 * mm, 105 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174EA6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1e293b")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f1f5f9")],
                    ),
                    ("PADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table

    result = str(record["prediction"])
    is_approved = result == "Approved"
    result_color = (
        colors.HexColor("#15803d")
        if is_approved
        else colors.HexColor("#dc2626")
    )

    story = []

    # Header
    story.append(Paragraph("LOANPREDICT AI", title_style))
    story.append(
        Paragraph("LOAN APPLICATION ASSESSMENT REPORT", subtitle_style)
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563eb"),
        )
    )
    story.append(Spacer(1, 15))

    # Application Information
    story.append(Paragraph("Application Information", section_style))

    application_data = [
        ["Application ID", str(record["id"])],
        ["Prediction Date", str(record["created_at"])],
        ["Application Status", result],
    ]

    application_table = Table(
        application_data,
        colWidths=[55 * mm, 115 * mm],
    )
    application_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf1fb")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(application_table)

    # Personal Information — append the table immediately after its heading.
    story.append(Paragraph("Personal Information", section_style))
    personal_data = [
        ["Field", "Applicant Information"],
        ["Gender", str(record["gender"])],
        ["Marital Status", str(record["married"])],
        ["Dependents", str(record["dependents"])],
        ["Education", str(record["education"])],
        ["Self Employed", str(record["self_employed"])],
        ["Property Area", str(record["property_area"])],
    ]
    story.append(create_table(personal_data))

    # Financial Information — append the table immediately after its heading.
    story.append(Paragraph("Financial Information", section_style))
    financial_data = [
        ["Field", "Value"],
        ["Applicant Income", f'{float(record["applicant_income"]):,.2f}'],
        ["Co-applicant Income", f'{float(record["coapplicant_income"]):,.2f}'],
        ["Loan Amount (Dataset Units)", f'{float(record["loan_amount"]):,.2f}'],
        ["Loan Term", f'{record["loan_term"]} months'],
    ]
    story.append(create_table(financial_data))

    # Credit Information — append the table immediately after its heading.
    story.append(Paragraph("Credit Information", section_style))
    credit_data = [
        ["Field", "Value"],
        [
            "Credit History",
            "Good (1)" if int(record["credit_history"]) == 1 else "Poor (0)",
        ],
    ]
    story.append(create_table(credit_data))

    # Prediction Result
    story.append(Paragraph("Machine Learning Prediction", section_style))

    result_text = (
        "LOAN LIKELY APPROVED" if is_approved else "LOAN LIKELY REJECTED"
    )
    result_style = ParagraphStyle(
        "Result",
        parent=normal_style,
        fontSize=16,
        leading=22,
        textColor=result_color,
        alignment=TA_CENTER,
    )

    result_table = Table(
        [[Paragraph(result_text, result_style)]],
        colWidths=[170 * mm],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 1, result_color),
                ("PADDING", (0, 0), (-1, -1), 15),
            ]
        )
    )
    story.append(result_table)

    # Model Scores
    story.append(Spacer(1, 12))
    score_data = [
        ["Model Score", "Percentage"],
        ["Approval Score", f'{float(record["approval_score"]) * 100:.2f}%'],
        ["Rejection Score", f'{float(record["rejection_score"]) * 100:.2f}%'],
    ]
    story.append(create_table(score_data))

    # Disclaimer and footer
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an academic "
            "Machine Learning application. Predictions are estimates based "
            "on the trained model and do not represent an actual bank "
            "decision, loan sanction, or guaranteed approval.",
            normal_style,
        )
    )
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Generated by LoanPredict AI | Machine Learning Academic Project",
            subtitle_style,
        )
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()

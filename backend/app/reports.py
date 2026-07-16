from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlmodel import Session, select

from .models_db import Patient, Prediction, Study, StudyImage

DISCLAIMER = (
    "AI Draft — Requires Clinician Review. This report was generated automatically "
    "by MedChron AI and has not been reviewed by a licensed clinician. It is not a "
    "diagnosis and must not be used for clinical decision-making without physician review."
)

MODALITY_LABELS = {
    "chest_xray": "Chest X-ray",
    "brain_mri": "Brain MRI",
    "mammography": "Mammography",
}


def build_report_data(study: Study, session: Session) -> dict:
    patient = session.get(Patient, study.patient_id) if study.patient_id else None
    pred = session.exec(select(Prediction).where(Prediction.study_id == study.id)).first()

    reasons = [r.strip() for r in (study.quality_reasons or "").split(";") if r.strip()]

    return {
        "study_id": study.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modality": study.modality,
        "modality_label": MODALITY_LABELS.get(study.modality, study.modality),
        "uploaded_at": study.uploaded_at.isoformat(),
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "sex": patient.sex,
            "birth_year": patient.birth_year,
        } if patient else None,
        "quality": {
            "passed": study.quality_passed,
            "score": study.quality_score,
            "reasons": reasons,
        },
        "num_rois": study.num_rois,
        "analysis_stopped": bool(study.analysis_stopped),
        "model_version": study.model_version,
        "processing_time_ms": study.processing_time_ms,
        "inference_time_ms": study.inference_time_ms,
        "prediction": {
            "label": pred.label,
            "confidence": pred.confidence,
            "probabilities": json.loads(pred.probabilities or "{}"),
            "backbone": pred.backbone,
            "per_model": json.loads(pred.per_model) if pred.per_model else None,
        } if pred else None,
        "disclaimer": DISCLAIMER,
    }


def _fetch_image_bytes(session: Session, study_id: int, name: str) -> Optional[bytes]:
    img = session.exec(
        select(StudyImage).where(StudyImage.study_id == study_id, StudyImage.name == name)
    ).first()
    return img.data if img else None


def render_report_pdf(report: dict, session: Session) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=18, spaceAfter=2)
    sub_style = ParagraphStyle("ReportSub", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    section_style = ParagraphStyle("Section", parent=styles["Heading3"], spaceBefore=14, spaceAfter=6)
    body_style = styles["Normal"]
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#8a5a00"),
        backColor=colors.HexColor("#fff6e0"), borderColor=colors.HexColor("#e0b04a"),
        borderWidth=0.75, borderPadding=8, spaceBefore=10, spaceAfter=10,
    )

    story = []
    story.append(Paragraph("MedChron AI — Imaging Report", title_style))
    story.append(Paragraph(
        f"Study #{report['study_id']} · {report['modality_label']} · "
        f"generated {report['generated_at'][:19].replace('T', ' ')} UTC",
        sub_style,
    ))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    patient = report["patient"]
    story.append(Paragraph("Patient", section_style))
    if patient:
        rows = [
            ["Name", patient["name"]],
            ["Sex", patient.get("sex") or "—"],
            ["Birth year", str(patient.get("birth_year") or "—")],
        ]
    else:
        rows = [["Name", "Not attached to a patient record"]]
    story.append(_kv_table(rows))

    story.append(Paragraph("Study", section_style))
    story.append(_kv_table([
        ["Modality", report["modality_label"]],
        ["Uploaded", report["uploaded_at"][:19].replace("T", " ")],
        ["Model version", report.get("model_version") or "—"],
    ]))

    story.append(Paragraph("Quality assessment", section_style))
    q = report["quality"]
    q_rows = [
        ["Passed", "Yes" if q["passed"] else "No"],
        ["Score", f"{q['score']}/100" if q["score"] is not None else "—"],
    ]
    story.append(_kv_table(q_rows))
    if q["reasons"]:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Reasons flagged: " + "; ".join(q["reasons"]), body_style))

    story.append(Paragraph("AI findings", section_style))
    if report["analysis_stopped"]:
        story.append(Paragraph(
            "Analysis was stopped before inference (image failed the quality gate). "
            "No prediction was produced.", body_style,
        ))
    elif report["prediction"]:
        pred = report["prediction"]
        story.append(_kv_table([
            ["Predicted label", pred["label"]],
            ["Confidence", f"{pred['confidence'] * 100:.1f}%"],
            ["Backbone", pred.get("backbone") or "—"],
        ]))
        if pred["probabilities"]:
            story.append(Spacer(1, 6))
            prob_rows = [["Class", "Probability"]] + [
                [k, f"{v * 100:.1f}%"] for k, v in sorted(
                    pred["probabilities"].items(), key=lambda kv: -kv[1]
                )
            ]
            story.append(_kv_table(prob_rows, header=True))
        if pred.get("per_model"):
            story.append(Spacer(1, 8))
            story.append(Paragraph("Model agreement (ensemble members)", sub_style))
            member_rows = [["Backbone", "Predicted", "Confidence"]] + [
                [m["backbone"], m["label"], f"{m['confidence'] * 100:.1f}%"]
                for m in pred["per_model"]
            ]
            story.append(_kv_table(member_rows, header=True))
    else:
        story.append(Paragraph("No prediction available for this study.", body_style))

    story.append(Paragraph(f"Regions of interest detected: {report['num_rois']}", section_style))

    original = _fetch_image_bytes(session, report["study_id"], "original")
    annotated = _fetch_image_bytes(session, report["study_id"], "rois")
    heatmap = _fetch_image_bytes(session, report["study_id"], "gradcam")
    images = [(label, data) for label, data in
              [("Original", original), ("ROI overlay", annotated), ("Grad-CAM", heatmap)] if data]
    if images:
        cells = []
        for label, data in images:
            img_reader = ImageReader(BytesIO(data))
            iw, ih = img_reader.getSize()
            target_w = 1.9 * inch
            target_h = target_w * ih / iw
            rl_img = RLImage(BytesIO(data), width=target_w, height=target_h)
            cells.append([rl_img, Paragraph(label, sub_style)])
        img_table = Table(
            [[c[0] for c in cells], [c[1] for c in cells]],
            colWidths=[2.1 * inch] * len(cells),
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(KeepTogether([Paragraph("Imaging", section_style), img_table]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    doc.build(story)
    return buf.getvalue()


def _kv_table(rows: list, header: bool = False) -> Table:
    t = Table(rows, colWidths=[2.1 * inch, 4.2 * inch])
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e5e5")),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
        ]
    else:
        style.append(("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t

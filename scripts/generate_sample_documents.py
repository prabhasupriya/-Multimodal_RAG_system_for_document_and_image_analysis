"""
Generates the `sample_documents/` collection used for testing and
evaluation: plain text files, standalone chart/diagram images, and
PDFs that mix text, tables, and embedded images.

Run:  python scripts/generate_sample_documents.py
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_documents"
TEXT_DIR = SAMPLES / "text"
IMG_DIR = SAMPLES / "images"
PDF_DIR = SAMPLES / "pdfs"


def _font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_bar_chart_image(path: Path, title: str, labels, values, ylabel="Value"):
    W, H = 640, 420
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), title, fill="black", font=_font(20))
    margin_l, margin_b, top = 70, 60, 60
    chart_h = H - top - margin_b
    max_v = max(values) * 1.2
    bar_w = (W - margin_l - 40) / len(values)
    for i, (label, val) in enumerate(zip(labels, values)):
        bar_h = int((val / max_v) * chart_h)
        x0 = margin_l + i * bar_w + 10
        x1 = x0 + bar_w - 20
        y1 = H - margin_b
        y0 = y1 - bar_h
        draw.rectangle([x0, y0, x1, y1], fill=(70, 110, 200))
        draw.text((x0, y0 - 20), str(val), fill="black", font=_font(14))
        draw.text((x0, y1 + 8), label, fill="black", font=_font(14))
    draw.line([(margin_l, top), (margin_l, H - margin_b)], fill="black", width=2)
    draw.line([(margin_l, H - margin_b), (W - 20, H - margin_b)], fill="black", width=2)
    draw.text((15, top - 30), ylabel, fill="black", font=_font(14))
    img.save(path)


def make_line_chart_image(path: Path, title: str, x_labels, series, legend_label):
    W, H = 640, 420
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), title, fill="black", font=_font(20))
    margin_l, margin_b, top = 70, 60, 60
    chart_w, chart_h = W - margin_l - 40, H - top - margin_b
    max_v = max(series) * 1.2
    min_v = 0
    step_x = chart_w / (len(series) - 1)
    points = []
    for i, v in enumerate(series):
        x = margin_l + i * step_x
        y = H - margin_b - ((v - min_v) / (max_v - min_v)) * chart_h
        points.append((x, y))
    draw.line(points, fill=(200, 70, 70), width=3)
    for (x, y), lbl in zip(points, x_labels):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(200, 70, 70))
        draw.text((x - 10, H - margin_b + 8), lbl, fill="black", font=_font(13))
    draw.line([(margin_l, top), (margin_l, H - margin_b)], fill="black", width=2)
    draw.line([(margin_l, H - margin_b), (W - 20, H - margin_b)], fill="black", width=2)
    draw.text((W - 220, top), f"— {legend_label}", fill=(200, 70, 70), font=_font(14))
    img.save(path)


def make_architecture_diagram(path: Path):
    W, H = 700, 300
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    boxes = [("Ingestion", 20), ("Embedding", 190), ("Vector Store", 360), ("Generation", 530)]
    y0, y1 = 110, 190
    for label, x in boxes:
        draw.rectangle([x, y0, x + 140, y1], outline="black", width=2, fill=(230, 240, 255))
        draw.text((x + 15, y0 + 30), label, fill="black", font=_font(16))
    for _, x in boxes[:-1]:
        draw.line([(x + 140, 150), (x + 190, 150)], fill="black", width=3)
        draw.polygon([(x + 190, 145), (x + 190, 155), (x + 200, 150)], fill="black")
    draw.text((20, 30), "Data Flow: Multimodal RAG Pipeline", fill="black", font=_font(20))
    img.save(path)


def pil_image_to_reader(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def make_pdf_report(path: Path, title: str, chart_img_path: Path, table_rows, body_paragraphs, page2_img_path=None):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Page 1: title + body text + table
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1 * inch, height - 1 * inch, title)
    c.setFont("Helvetica", 11)
    y = height - 1.4 * inch
    for para in body_paragraphs:
        for line in _wrap(para, 95):
            c.drawString(1 * inch, y, line)
            y -= 14
        y -= 8

    # Table
    table = Table(table_rows, colWidths=[110] * len(table_rows[0]))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4466CC")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    table.wrapOn(c, width, height)
    table.drawOn(c, 1 * inch, y - 20 - 18 * len(table_rows))
    c.showPage()

    # Page 2: embedded chart image
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Figure: Supporting Chart")
    c.drawImage(str(chart_img_path), 1 * inch, height - 5.2 * inch, width=5.5 * inch, height=3.6 * inch)
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 5.5 * inch, "The chart above visualizes the key figures referenced on page 1.")
    if page2_img_path:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1 * inch, height - 1 * inch, "Figure: Additional Diagram")
        c.drawImage(str(page2_img_path), 1 * inch, height - 4.5 * inch, width=5.8 * inch, height=2.5 * inch)
    c.save()


def _wrap(text, width):
    import textwrap

    return textwrap.wrap(text, width) or [""]


def main():
    for d in (TEXT_DIR, IMG_DIR, PDF_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1-3: plain text files ----
    (TEXT_DIR / "company_overview.txt").write_text(
        "Acme Robotics Inc. Company Overview\n\n"
        "Acme Robotics was founded in 2016 and specializes in autonomous "
        "warehouse robots. The company's flagship product, the AcmeBot X1, "
        "is deployed in over 200 warehouses worldwide.\n\n"
        "In fiscal year 2023, Acme Robotics reported total revenue of $48 "
        "million, a 32% increase over the prior year. The increase was "
        "driven primarily by strong demand for the X1 in the logistics "
        "sector, particularly in North America and Western Europe.\n\n"
        "The engineering team is organized into three groups: Perception, "
        "Controls, and Fleet Software. The Perception group is responsible "
        "for the computer vision stack that lets the X1 navigate cluttered "
        "warehouse aisles without human intervention.",
        encoding="utf-8",
    )
    (TEXT_DIR / "product_faq.txt").write_text(
        "AcmeBot X1 - Frequently Asked Questions\n\n"
        "Q: What is the maximum payload of the AcmeBot X1?\n"
        "A: The X1 can carry up to 500 kilograms on a standard pallet.\n\n"
        "Q: How long does the battery last?\n"
        "A: A full charge provides approximately 10 hours of continuous "
        "operation under typical warehouse loads.\n\n"
        "Q: Is the X1 compatible with existing warehouse management systems?\n"
        "A: Yes. The X1 integrates via a REST API with most major WMS "
        "platforms, including SAP EWM and Manhattan Associates.",
        encoding="utf-8",
    )
    (TEXT_DIR / "safety_policy.txt").write_text(
        "Warehouse Robot Safety Policy\n\n"
        "All autonomous mobile robots (AMRs) operating on the warehouse "
        "floor must maintain a minimum clearance of 0.5 meters from "
        "pedestrian walkways. Robots must reduce speed to under 0.3 m/s "
        "whenever a human is detected within 2 meters.\n\n"
        "Emergency stop buttons must be present on all AMR units and "
        "tested weekly. Any safety incident, including near-misses, must "
        "be logged in the incident tracker within 24 hours.",
        encoding="utf-8",
    )

    # ---- 4-6: standalone chart / diagram images (with OCR-able text) ----
    make_bar_chart_image(
        IMG_DIR / "quarterly_sales_chart.png",
        "Quarterly Sales (2023, $M)",
        ["Q1", "Q2", "Q3", "Q4"],
        [9.5, 10.8, 12.0, 15.7],
        ylabel="Revenue ($M)",
    )
    make_line_chart_image(
        IMG_DIR / "battery_life_trend.png",
        "AcmeBot X1 Battery Life Over Generations",
        ["Gen1", "Gen2", "Gen3", "Gen4"],
        [6, 7.5, 8.8, 10],
        legend_label="Hours per charge",
    )
    make_architecture_diagram(IMG_DIR / "system_architecture_diagram.png")

    # ---- 7-10: PDFs mixing text, tables, and images ----
    make_pdf_report(
        PDF_DIR / "annual_report_2023.pdf",
        "Acme Robotics — Annual Report 2023",
        IMG_DIR / "quarterly_sales_chart.png",
        table_rows=[
            ["Quarter", "Revenue ($M)", "Units Shipped", "Gross Margin"],
            ["Q1", "9.5", "120", "41%"],
            ["Q2", "10.8", "138", "42%"],
            ["Q3", "12.0", "151", "44%"],
            ["Q4", "15.7", "190", "46%"],
        ],
        body_paragraphs=[
            "Acme Robotics closed fiscal year 2023 with total revenue of $48.0 million, "
            "up 32% year over year. Growth accelerated through the year, with Q4 "
            "revenue of $15.7 million representing the strongest quarter in company history.",
            "Gross margin expanded from 41% in Q1 to 46% in Q4, driven by manufacturing "
            "efficiencies at the company's new Ohio assembly facility.",
        ],
    )
    make_pdf_report(
        PDF_DIR / "product_spec_x1.pdf",
        "AcmeBot X1 — Technical Specification",
        IMG_DIR / "battery_life_trend.png",
        table_rows=[
            ["Spec", "Value"],
            ["Max payload", "500 kg"],
            ["Battery life", "10 hours"],
            ["Top speed", "1.8 m/s"],
            ["Navigation", "LiDAR + stereo vision"],
        ],
        body_paragraphs=[
            "The AcmeBot X1 is a fourth-generation autonomous mobile robot designed for "
            "high-throughput warehouse environments. Battery life has improved with each "
            "generation, from 6 hours in Gen1 to 10 hours in the current Gen4 model.",
        ],
    )
    make_pdf_report(
        PDF_DIR / "system_architecture_whitepaper.pdf",
        "Multimodal RAG Reference Architecture",
        IMG_DIR / "system_architecture_diagram.png",
        table_rows=[
            ["Stage", "Primary Technology"],
            ["Ingestion", "PyMuPDF, pdfplumber, pytesseract"],
            ["Embedding", "CLIP (sentence-transformers)"],
            ["Vector Store", "ChromaDB"],
            ["Generation", "GPT-4V-class VLM"],
        ],
        body_paragraphs=[
            "This whitepaper describes a reference architecture for retrieval-augmented "
            "generation over multimodal document collections. The pipeline is organized "
            "into five decoupled stages: ingestion, embedding, indexing, retrieval, and "
            "generation, as illustrated in the architecture diagram on the following page.",
        ],
        page2_img_path=IMG_DIR / "system_architecture_diagram.png",
    )
    make_pdf_report(
        PDF_DIR / "safety_incident_summary.pdf",
        "Warehouse Safety Incident Summary — 2023",
        IMG_DIR / "quarterly_sales_chart.png",
        table_rows=[
            ["Quarter", "Near-Misses", "Incidents", "Avg. Resolution (days)"],
            ["Q1", "4", "0", "2"],
            ["Q2", "3", "1", "3"],
            ["Q3", "2", "0", "1"],
            ["Q4", "1", "0", "1"],
        ],
        body_paragraphs=[
            "Safety near-misses declined steadily throughout 2023, from 4 in Q1 to 1 in "
            "Q4, following the rollout of the updated pedestrian-detection firmware in "
            "Q2. No incidents resulted in injury.",
        ],
    )

    # ---- 11: extra standalone image (diversity / safety margin above the min of 10) ----
    make_bar_chart_image(
        IMG_DIR / "regional_deployment_chart.png",
        "AcmeBot X1 Deployments by Region (2023)",
        ["N.America", "Europe", "APAC"],
        [120, 58, 22],
        ylabel="Units deployed",
    )

    # ---- 12: extra plain text file ----
    (TEXT_DIR / "customer_testimonials.txt").write_text(
        "AcmeBot X1 - Customer Testimonials\n\n"
        "\"Since deploying twelve AcmeBot X1 units in our Ohio distribution "
        "center, order picking errors have dropped by 18%.\" - Fulfillment "
        "Director, MidState Logistics.\n\n"
        "\"The 10-hour battery life means we run a full shift without a "
        "single mid-day charging break.\" - Operations Manager, "
        "Northbridge Retail.\n\n"
        "\"Integration with our WMS took under a week thanks to the REST "
        "API.\" - IT Lead, Harborview Distribution.",
        encoding="utf-8",
    )

    # ---- 13: extra PDF with a competitor comparison table + chart (diversity) ----
    make_pdf_report(
        PDF_DIR / "competitor_comparison.pdf",
        "Warehouse AMR Market — Competitive Comparison",
        IMG_DIR / "regional_deployment_chart.png",
        table_rows=[
            ["Robot", "Max Payload", "Battery Life", "Price Tier"],
            ["AcmeBot X1", "500 kg", "10 hours", "Mid"],
            ["Competitor A", "400 kg", "8 hours", "Mid"],
            ["Competitor B", "600 kg", "6 hours", "High"],
        ],
        body_paragraphs=[
            "Among mid-tier autonomous mobile robots, the AcmeBot X1 offers the "
            "longest battery life at 10 hours per charge, while matching or "
            "exceeding the payload capacity of similarly priced competitors.",
            "Regional deployment data shows North America remains the largest "
            "market for the X1, followed by Europe and Asia-Pacific.",
        ],
    )

    print("Sample documents generated:")
    for f in sorted(SAMPLES.rglob("*")):
        if f.is_file():
            print(" -", f.relative_to(SAMPLES))


if __name__ == "__main__":
    main()

"""
Generates a small set of synthetic images (bar chart, line chart, pie
chart, table, and a scanned-document-style image) with known ground-truth
values baked in, alongside a reference_qa.json of questions whose answers
are objectively checkable against those values.

Synthetic data instead of scraped/downloaded images on purpose: every
answer here is verifiable by construction (the bar chart's Q3 value is
exactly 90 because that's what was plotted), which makes grading in
evaluation/ unambiguous -- no need to guess what the "real" answer to a
found-online chart was.

Usage:
    python generate_sample_images.py --output_dir ../data/images
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def generate_bar_chart(output_dir: Path) -> dict:
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    revenue = [120, 150, 90, 200]  # in $thousands

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(quarters, revenue, color="#4C72B0")
    ax.set_title("Quarterly Revenue ($ thousands)")
    ax.set_ylabel("Revenue ($k)")
    fig.tight_layout()

    path = output_dir / "bar_chart_revenue.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "image": path.name,
        "image_type": "chart",
        "questions": [
            {"question": "What was the revenue in Q3?", "reference_answer": "$90k (90 thousand dollars)"},
            {"question": "Which quarter had the highest revenue?", "reference_answer": "Q4"},
            {"question": "What was the total revenue across all four quarters?", "reference_answer": "$560k"},
        ],
    }


def generate_line_chart(output_dir: Path) -> dict:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    traffic = [1000, 1200, 900, 1500, 1700, 1400]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(months, traffic, marker="o", color="#DD8452")
    ax.set_title("Monthly Website Traffic")
    ax.set_ylabel("Visitors")
    fig.tight_layout()

    path = output_dir / "line_chart_traffic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "image": path.name,
        "image_type": "chart",
        "questions": [
            {"question": "In which month was traffic highest?", "reference_answer": "May"},
            {"question": "Did traffic increase or decrease from February to March?", "reference_answer": "decreased"},
            {"question": "What was the traffic in April?", "reference_answer": "1500"},
        ],
    }


def generate_pie_chart(output_dir: Path) -> dict:
    companies = ["Company A", "Company B", "Company C", "Company D"]
    share = [40, 25, 20, 15]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(share, labels=companies, autopct="%1.0f%%", colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_title("Market Share by Company")
    fig.tight_layout()

    path = output_dir / "pie_chart_market_share.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "image": path.name,
        "image_type": "chart",
        "questions": [
            {"question": "Which company has the largest market share?", "reference_answer": "Company A"},
            {"question": "What percentage of the market does Company C have?", "reference_answer": "20%"},
            {"question": "Which company has the smallest market share?", "reference_answer": "Company D"},
        ],
    }


def generate_table_image(output_dir: Path) -> dict:
    rows = [
        ("Product", "Price", "Stock"),
        ("Widget A", "$10", "50"),
        ("Widget B", "$25", "12"),
        ("Widget C", "$5", "200"),
    ]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    fig.tight_layout()

    path = output_dir / "table_inventory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "image": path.name,
        "image_type": "table",
        "questions": [
            {"question": "How many Widget B are in stock?", "reference_answer": "12"},
            {"question": "Which product is the cheapest?", "reference_answer": "Widget C"},
            {"question": "What is the price of Widget A?", "reference_answer": "$10"},
        ],
    }


def generate_document_image(output_dir: Path) -> dict:
    text_lines = [
        "MEETING NOTICE",
        "",
        "Project: Q3 Roadmap Review",
        "Date: March 15th",
        "Time: 2:00 PM",
        "Location: Room 204",
        "",
        "Please bring your team's status updates.",
        "Contact: Sarah Chen, ext. 4471",
    ]

    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)
    font = load_font(20)

    y = 30
    for line in text_lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 35

    path = output_dir / "document_meeting_notice.png"
    img.save(path)

    return {
        "image": path.name,
        "image_type": "document",
        "questions": [
            {"question": "What room is the meeting in?", "reference_answer": "Room 204"},
            {"question": "What time is the meeting?", "reference_answer": "2:00 PM"},
            {"question": "Who should be contacted, and at what extension?", "reference_answer": "Sarah Chen, ext. 4471"},
        ],
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        generate_bar_chart,
        generate_line_chart,
        generate_pie_chart,
        generate_table_image,
        generate_document_image,
    ]

    dataset = []
    for generator in generators:
        entry = generator(output_dir)
        dataset.append(entry)
        print(f"Generated {entry['image']} ({len(entry['questions'])} questions)")

    qa_path = Path(args.output_dir).parent / "reference_qa.json"
    with open(qa_path, "w") as f:
        json.dump(dataset, f, indent=2)

    total_questions = sum(len(e["questions"]) for e in dataset)
    print(f"\nGenerated {len(dataset)} images, {total_questions} reference questions.")
    print(f"Images: {output_dir}")
    print(f"Reference Q&A: {qa_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chart/table/document images with reference Q&A")
    parser.add_argument("--output_dir", default="../data/images")
    args = parser.parse_args()

    main(args)

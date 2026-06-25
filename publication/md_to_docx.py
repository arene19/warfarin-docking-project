#!/usr/bin/env python3
"""Convert manuscript_draft.md to manuscript_draft.docx with embedded figures."""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "manuscript_draft.md"
DOCX_PATH = ROOT / "manuscript_draft.docx"

FIG_PATH_RE = re.compile(r"`((?:publication/)?output/figures/[^`]+\.png)`")
FIG_LABEL_RE = re.compile(
    r"\*\*(Figure\s+(?:S)?\d+|Morgan standalone)\*\*",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text


def add_heading(doc: Document, text: str, level: int) -> None:
    text = clean(text)
    if text:
        doc.add_heading(text, level=min(level, 3))


def resolve_figure_path(path_str: str) -> Path | None:
    candidates = [
        ROOT / path_str,
        ROOT / path_str.replace("publication/", ""),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def embed_figure(doc: Document, path_str: str, caption: str | None = None) -> bool:
    path = resolve_figure_path(path_str)
    if path is None:
        p = doc.add_paragraph(f"[Figure not found: {path_str}]")
        p.runs[0].italic = True
        return False

    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.add_run().add_picture(str(path), width=Inches(6.0))

    if caption:
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.size = Pt(10)
    return True


def figure_caption_from_line(line: str, path_str: str) -> str:
    label_match = FIG_LABEL_RE.search(line)
    label = label_match.group(1) if label_match else Path(path_str).stem.replace("_", " ")
    rest = line
    if label_match:
        rest = rest[label_match.end() :]
    rest = rest.replace(f"`{path_str}`", "")
    rest = clean(rest.lstrip(" :—-"))
    if rest:
        return f"{label}. {rest}"
    return label


def embed_figures_in_line(doc: Document, line: str, seen: set[str]) -> None:
    for match in FIG_PATH_RE.finditer(line):
        path_str = match.group(1)
        if path_str in seen:
            continue
        seen.add(path_str)
        caption = figure_caption_from_line(line, path_str)
        embed_figure(doc, path_str, caption)


def embed_figures_in_table(doc: Document, rows: list[list[str]], seen: set[str]) -> None:
    for row in rows:
        row_text = " | ".join(row)
        for match in FIG_PATH_RE.finditer(row_text):
            path_str = match.group(1)
            if path_str in seen:
                continue
            seen.add(path_str)
            label = clean(row[0]) if row else path_str
            embed_figure(doc, path_str, label)


def main() -> None:
    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    in_code = False
    code_lines: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    seen_figures: set[str] = set()

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        tbl = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        tbl.style = "Table Grid"
        for i, row in enumerate(table_rows):
            for j, cell in enumerate(row):
                tbl.rows[i].cells[j].text = clean(cell)
        embed_figures_in_table(doc, table_rows, seen_figures)
        table_rows = []
        in_table = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                p = doc.add_paragraph("\n".join(code_lines))
                for run in p.runs:
                    run.font.name = "Courier New"
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if line.strip().startswith("|") and "|" in line[1:]:
            if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            continue
        if in_table:
            flush_table()

        if line.startswith("# "):
            add_heading(doc, line[2:], 0)
        elif line.startswith("## "):
            add_heading(doc, line[3:], 1)
        elif line.startswith("### "):
            add_heading(doc, line[4:], 2)
        elif line.startswith("#### "):
            add_heading(doc, line[5:], 3)
        elif line.strip() == "---":
            doc.add_paragraph()
        elif line.startswith("> "):
            p = doc.add_paragraph(clean(line[2:]))
            if p.runs:
                p.runs[0].italic = True
        elif line.startswith("- "):
            doc.add_paragraph(clean(line[2:]), style="List Bullet")
        elif re.match(r"^\d+\.\s", line):
            doc.add_paragraph(clean(re.sub(r"^\d+\.\s", "", line)), style="List Number")
        elif line.strip().startswith("$$"):
            doc.add_paragraph(clean(line.strip("$")))
        elif line.strip():
            doc.add_paragraph(clean(line))
            embed_figures_in_line(doc, line, seen_figures)
        else:
            doc.add_paragraph()

    if in_table:
        flush_table()

    doc.save(DOCX_PATH)
    print(f"Wrote {DOCX_PATH} ({len(seen_figures)} figures embedded)")


if __name__ == "__main__":
    main()

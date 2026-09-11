"""Generate a single-column, Word-style PDF from paper.tex -- i.e. a PDF
rendering of the SAME content as paper.docx, as opposed to paper.pdf,
which is the two-column IEEE conference layout.

No installed copy of Word or LibreOffice is available in this environment
to export paper.docx to PDF directly, so this takes the same approach as
make_paper_docx.py: parse paper.tex directly (our own controlled-vocabulary
LaTeX) and render it, this time with reportlab instead of python-docx, so
both output formats trace to the same source rather than one being a
lossy export of the other.

Reuses every text-cleaning/citation/label helper from make_paper_docx.py
unchanged -- only the rendering backend differs (reportlab flowables
instead of docx paragraphs/runs), so both documents can never disagree
about what a citation number or a table caption says.

Usage:
    python ml/scripts/make_paper_word_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from make_paper_docx import (  # the already-verified LaTeX->plain-text helpers
    CAPTION_LABELS,
    MATH_SUBS,
    PAPER_DIR,
    REF_LABELS,
    TEX,
    load_bib_entries,
    load_bib_order,
    parse_tex_table,
)

OUT = PAPER_DIR / "paper_word.pdf"

FONTS_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TimesNR", str(FONTS_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("TimesNR-Bold", str(FONTS_DIR / "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("TimesNR-Italic", str(FONTS_DIR / "timesi.ttf")))
pdfmetrics.registerFont(TTFont("TimesNR-BoldItalic", str(FONTS_DIR / "timesbi.ttf")))
pdfmetrics.registerFontFamily(
    "TimesNR", normal="TimesNR", bold="TimesNR-Bold",
    italic="TimesNR-Italic", boldItalic="TimesNR-BoldItalic",
)

# reportlab's Paragraph markup does not do real text shaping, so a
# combining diacritic (e.g. U+0304 COMBINING MACRON for x-bar) would not
# visually overlay its base character -- it would render as a stray mark
# next to it. Spell it out instead; this is the only entry that differs
# from the .docx version's MATH_SUBS, and it changes no factual content.
PDF_MATH_SUBS = [(p, r) if p != r"\\bar\{x\}" else (p, "x-bar") for p, r in MATH_SUBS]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clean_plain(s: str) -> str:
    s = s.replace("{,}", ",")
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    for pat, lit in (("\\%", "%"), ("\\_", "_"), ("\\&", "&"), ("\\#", "#"), ("\\$", "$")):
        s = s.replace(pat, lit)
    s = s.replace("~", " ")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    return s


def inline_to_plain(s: str) -> str:
    for pat, rep in PDF_MATH_SUBS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\cite\{[^}]+\}", "", s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: CAPTION_LABELS.get(m.group(1), m.group(1)), s)
    s = s.replace("{,}", ",")
    s = s.replace("\\{", "\x00OB\x00").replace("\\}", "\x00CB\x00")
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    for pat, lit in (("\\%", "%"), ("\\_", "_"), ("\\&", "&"), ("\\#", "#"), ("\\$", "$")):
        s = s.replace(pat, lit)
    s = s.replace("~", " ")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    s = s.replace("\x00OB\x00", "{").replace("\x00CB\x00", "}")
    return re.sub(r"\s+", " ", s).strip()


def to_markup(text: str, bib_order: dict) -> str:
    """LaTeX paragraph text -> reportlab Paragraph markup (<i>/<b> spans),
    the PDF-rendering equivalent of make_paper_docx.add_runs()."""
    token_re = re.compile(
        r"\\emph\{(?P<emph>[^}]*)\}"
        r"|\$(?P<math>[^$]*)\$"
        r"|\\cite\{(?P<cite>[^}]+)\}"
        r"|\\ref\{(?P<ref>[^}]+)\}"
    )
    out, pos = [], 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            out.append(esc(_clean_plain(text[pos:m.start()])))
        if m.group("emph") is not None:
            out.append(f"<i>{esc(_clean_plain(m.group('emph')))}</i>")
        elif m.group("math") is not None:
            content = m.group("math")
            for pat, rep in PDF_MATH_SUBS:
                content = re.sub(pat, rep, content)
            content = re.sub(r"[{}\\]", "", content)
            out.append(f"<i>{esc(content)}</i>")
        elif m.group("cite") is not None:
            keys = [k.strip() for k in m.group("cite").split(",")]
            nums = [str(bib_order[k]) for k in keys]
            out.append(f"[{','.join(nums)}]")
        elif m.group("ref") is not None:
            out.append(esc(REF_LABELS.get(m.group("ref"), m.group("ref"))))
        pos = m.end()
    if pos < len(text):
        out.append(esc(_clean_plain(text[pos:])))
    return "".join(out)


# ---- styles -----------------------------------------------------------
BASE = dict(fontName="TimesNR", fontSize=11, leading=15)
STYLES = {
    "title": ParagraphStyle("title", fontName="TimesNR-Bold", fontSize=18, leading=22,
                             alignment=TA_CENTER, spaceAfter=10),
    "byline": ParagraphStyle("byline", fontName="TimesNR", fontSize=11, leading=14,
                              alignment=TA_CENTER, spaceAfter=16),
    "abstract": ParagraphStyle("abstract", **BASE, alignment=TA_JUSTIFY,
                                leftIndent=18, rightIndent=18, spaceAfter=10),
    "keywords": ParagraphStyle("keywords", **BASE, leftIndent=18, rightIndent=18, spaceAfter=16),
    "h1": ParagraphStyle("h1", fontName="TimesNR-Bold", fontSize=14, leading=18,
                          spaceBefore=16, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="TimesNR-Bold", fontSize=12, leading=16,
                          spaceBefore=12, spaceAfter=6),
    "body": ParagraphStyle("body", **BASE, alignment=TA_JUSTIFY, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="TimesNR-Italic", fontSize=9.5, leading=12,
                               alignment=TA_CENTER, spaceBefore=4, spaceAfter=12),
    "ref": ParagraphStyle("ref", fontName="TimesNR", fontSize=10, leading=13,
                           leftIndent=18, spaceAfter=6),
}


def build_table(tex_data: dict, label: str) -> list:
    rows = tex_data["rows"]
    ncols = max(len(r) for r in rows)
    data = []
    for row in rows:
        cells = [inline_to_plain(row[j]) if j < len(row) else "" for j in range(ncols)]
        data.append(cells)

    t = Table(data, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "TimesNR-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "TimesNR"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    cap = Paragraph(f"<b>{label}.</b> {esc(tex_data['caption'])}", STYLES["caption"])
    return [t, Spacer(1, 4), cap]


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    bib_order = load_bib_order(text)

    title = re.search(r"\\title\{(.+?)\}\s*\n", text, re.S).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", text, re.S).group(1).strip()
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.+?)\\end\{IEEEkeywords\}", text, re.S).group(1).strip()

    story = []
    story.append(Paragraph(to_markup(title, bib_order), STYLES["title"]))
    story.append(Paragraph(
        "Aditya Ayushman Sahoo<br/>"
        "Independent Researcher \u2014 please replace with your institution before submission<br/>"
        "adityaasahoo@gmail.com",
        STYLES["byline"],
    ))
    story.append(Paragraph(f"<b>Abstract\u2014</b><i>{to_markup(abstract, bib_order)}</i>", STYLES["abstract"]))
    story.append(Paragraph(f"<b>Index Terms\u2014</b><i>{esc(inline_to_plain(keywords))}</i>", STYLES["keywords"]))

    body = text.split("\\maketitle", 1)[1].split("\\begin{thebibliography}")[0]
    body = body.split("\\begin{IEEEkeywords}")[1].split("\\end{IEEEkeywords}", 1)[1]

    section_num = 0
    subsection_letter = 0
    roman = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        m = re.match(r"\\section\{(.+?)\}", block)
        if m:
            section_num += 1
            subsection_letter = 0
            story.append(Paragraph(f"{roman[section_num]}. {esc(inline_to_plain(m.group(1)))}", STYLES["h1"]))
            rest = re.sub(r"^\\label\{[^}]+\}", "", block[m.end():].strip()).strip()
            if rest:
                story.append(Paragraph(to_markup(rest, bib_order), STYLES["body"]))
            continue

        m = re.match(r"\\subsection\{(.+?)\}", block)
        if m:
            subsection_letter += 1
            letter = chr(ord("A") + subsection_letter - 1)
            story.append(Paragraph(f"{letter}. {esc(inline_to_plain(m.group(1)))}", STYLES["h2"]))
            rest = block[m.end():].strip()
            if rest:
                story.append(Paragraph(to_markup(rest, bib_order), STYLES["body"]))
            continue

        m = re.match(r"\\input\{tables/(.+?)\.tex\}", block)
        if m:
            tex_data = parse_tex_table(PAPER_DIR / "tables" / f"{m.group(1)}.tex")
            story.extend(build_table(tex_data, CAPTION_LABELS.get(tex_data["label"], "Table")))
            continue

        m = re.match(r"\\begin\{table\}.*?\\caption\{(.+?)\}\s*\\label\{([^}]+)\}\s*"
                     r"\\begin\{tabular\}\{[^}]*\}(.+?)\\end\{tabular\}\s*\\end\{table\}",
                     block, re.S)
        if m:
            caption, label_key, tbody = m.groups()
            rows = []
            for line in tbody.splitlines():
                line = line.strip()
                if not line or line.startswith(("\\toprule", "\\midrule", "\\bottomrule")):
                    continue
                line = line.rstrip("\\").strip()
                if line:
                    rows.append([c.strip() for c in line.split("&")])
            story.extend(build_table({"caption": inline_to_plain(caption), "rows": rows},
                                      CAPTION_LABELS.get(label_key, "Table")))
            continue

        m = re.match(
            r"\\begin\{figure\}.*?\\includegraphics\[[^\]]*\]\{figures/([^}]+)\.pdf\}"
            r"\s*\\caption\{(.+?)\}\s*\\label\{([^}]+)\}",
            block, re.S,
        )
        if m:
            fname, caption, label_key = m.groups()
            png = PAPER_DIR / "figures" / f"{fname}.png"
            img = Image(str(png), width=4.6 * inch, height=4.6 * inch * 0.62)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Paragraph(
                f"<b>{CAPTION_LABELS.get(label_key, 'Fig.')}.</b> {esc(inline_to_plain(caption))}",
                STYLES["caption"],
            ))
            continue

        story.append(Paragraph(to_markup(block, bib_order), STYLES["body"]))

    story.append(PageBreak())
    story.append(Paragraph("References", STYLES["h1"]))
    entries = load_bib_entries(text)
    items = [ListItem(Paragraph(esc(e), STYLES["ref"]), leftIndent=18) for e in entries]
    story.append(ListFlowable(items, bulletType="1", start=1))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=1 * inch, rightMargin=1 * inch, topMargin=1 * inch, bottomMargin=1 * inch,
        title="Beyond the Single Run",
    )
    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

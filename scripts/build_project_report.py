"""Build the portrait, source-backed Russian project report without API calls.

Requires ReportLab and Pillow. Source chapters, fonts and facts live in
docs/report_source and docs/report_assets. The app environment is independent.
"""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import re

from reportlab import rl_config
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, Table, TableStyle, Image, KeepTogether, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/report_assets"
SOURCE = ROOT / "docs/report_source"
INK = colors.HexColor("#1f2f22")
FOREST = colors.HexColor("#3f553e")
SAGE = colors.HexColor("#93a889")
BLUE = colors.HexColor("#7993a0")
PALE = colors.HexColor("#edf2ec")
LINE = colors.HexColor("#dce3dc")
MUTED = colors.HexColor("#637163")
PAGE_W, PAGE_H = A4
MARGIN = 53
WIDTH = PAGE_W - MARGIN * 2

def register_fonts():
    for name, file in [("Noto", "NotoSans-Regular.ttf"), ("Noto-Bold", "NotoSans-Bold.ttf"),
                       ("JetBrains", "JetBrainsMono-Medium.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(ASSETS / "fonts" / file)))
    pdfmetrics.registerFontFamily("Noto", normal="Noto", bold="Noto-Bold", italic="Noto", boldItalic="Noto-Bold")

def styles():
    base = dict(fontName="Noto", fontSize=10.8, leading=16.7, textColor=INK,
                spaceAfter=9, allowWidows=0, allowOrphans=0, splitLongWords=True)
    return {
        "body": ParagraphStyle("Body", **base),
        "h1": ParagraphStyle("Heading1", fontName="Noto-Bold", fontSize=23, leading=29,
                             textColor=FOREST, spaceAfter=18, keepWithNext=True),
        "h2": ParagraphStyle("Heading2", fontName="Noto-Bold", fontSize=14.3, leading=20,
                             textColor=FOREST, spaceBefore=15, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("Heading3", fontName="Noto-Bold", fontSize=11.7, leading=17,
                             textColor=FOREST, spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "bullet": ParagraphStyle("Bullet", **{**base, "leftIndent": 13, "firstLineIndent": -10, "spaceAfter": 5}),
        "caption": ParagraphStyle("Caption", fontName="Noto", fontSize=9, leading=13.3,
                                  textColor=MUTED, spaceBefore=6, spaceAfter=13),
        "refs": ParagraphStyle("References", fontName="Noto", fontSize=8.7, leading=12.3,
                               textColor=MUTED, spaceBefore=2, spaceAfter=2),
        "code": ParagraphStyle("Code", fontName="JetBrains", fontSize=8.3, leading=12.4,
                               textColor=INK, backColor=PALE, borderPadding=10, spaceBefore=6,
                               spaceAfter=14, splitLongWords=True),
        "cell": ParagraphStyle("Cell", fontName="Noto", fontSize=9.1, leading=13.4,
                               textColor=INK, spaceAfter=0),
        "thead": ParagraphStyle("TableHeader", fontName="Noto-Bold", fontSize=9.1, leading=13.4,
                                textColor=colors.white, spaceAfter=0),
        "toc": ParagraphStyle("Toc", fontName="Noto", fontSize=11.4, leading=22,
                              textColor=INK, leftIndent=0, firstLineIndent=0),
    }

def inline(text):
    text = text.replace("—", "-").replace("–", "-").replace("‑", "-")
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="JetBrains" size="9">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#3f553e"><u>\1</u></a>', text)
    return text

def wrap_lines(text, width, font="Noto", size=10):
    result = []
    for raw in text.split("\n"):
        line = ""
        for word in raw.split():
            trial = (line + " " + word).strip()
            if pdfmetrics.stringWidth(trial, font, size) > width and line:
                result.append(line)
                line = word
            else:
                line = trial
        result.append(line)
    return result

def diagram(kind):
    """Native, selectable technical diagrams and charts; no decorative artwork."""
    width = 480
    height = {"rag": 143, "architecture": 315, "indexing": 215, "chunks": 190,
              "dense": 160, "sign_links": 218, "api_boundary": 192,
              "quality": 180, "latency": 180, "atomic": 133}[kind]
    d = Drawing(width, height)
    def text(x, y, value, size=10, color=INK, font="Noto", anchor="start"):
        d.add(String(x, y, value, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))
    def node(x, y, w, h, label, detail="", fill=PALE):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=LINE, strokeWidth=.8))
        lines = wrap_lines(label, w - 18, "Noto-Bold", 10)
        for i, line in enumerate(lines):
            text(x + 9, y + h - 18 - i * 13, line, 10, font="Noto-Bold")
        if detail:
            for j, line in enumerate(wrap_lines(detail, w - 18, size=8.7)):
                text(x + 9, y + h - 23 - len(lines) * 13 - j * 11, line, 8.7, MUTED)
    def arrow(points):
        for a, b in zip(points, points[1:]):
            d.add(Line(*a, *b, strokeColor=FOREST, strokeWidth=1.1))
        (x0, y0), (x, y) = points[-2:]
        if abs(x - x0) >= abs(y - y0):
            delta = -5 if x > x0 else 5
            tri = [x, y, x + delta, y + 2.7, x + delta, y - 2.7]
        else:
            delta = -5 if y > y0 else 5
            tri = [x, y, x - 2.7, y + delta, x + 2.7, y + delta]
        d.add(Polygon(tri, fillColor=FOREST, strokeColor=None))
    if kind == "rag":
        for x, label, detail in [(0, "Вопрос", "Что означает знак?"),
                                  (168, "Найденные источники", "Текст из своей базы"),
                                  (336, "Ответ", "Модель объясняет текст")]:
            node(x, 48, 143, 77, label, detail)
        arrow([(143, 87), (166, 87)])
        arrow([(311, 87), (334, 87)])
        text(0, 17, "Поиск подбирает материал. Генератор формулирует объяснение.", 10)
    elif kind == "architecture":
        node(0, 235, 144, 68, "Вопрос в Streamlit", "app.py")
        node(168, 235, 144, 68, "BM25 / E5 / FAISS", "retrieval.py")
        node(336, 235, 144, 68, "Контекст ПДД", "traffic.py")
        arrow([(144, 267), (166, 267)])
        arrow([(312, 267), (334, 267)])
        node(336, 123, 144, 72, "Генератор через API", "generation.py")
        node(168, 123, 144, 72, "Картинки из файлов", "ID, путь, SHA-256")
        arrow([(408, 235), (408, 197)])
        arrow([(360, 235), (360, 215), (240, 215), (240, 197)])
        node(168, 13, 144, 69, "Ответ в чате", "Текст, изображения, ссылки")
        arrow([(240, 123), (240, 84)])
        arrow([(408, 123), (408, 99), (285, 99), (285, 84)])
        text(0, 148, "Laya необязательна", 9.5, MUTED)
        text(0, 133, "и работает в поиске", 9.5, MUTED)
    elif kind == "indexing":
        node(0, 143, 138, 62, "Исходные страницы", "HTML, таблицы, файлы")
        node(170, 143, 138, 62, "Документы", "Чистый текст и исходные ID")
        node(342, 143, 138, 62, "Чанки", "384 токена, перекрытие 64")
        arrow([(138, 173), (168, 173)])
        arrow([(308, 173), (340, 173)])
        node(170, 25, 138, 70, "BM25-индекс", "Поиск совпадений слов")
        node(342, 25, 138, 70, "E5 и FAISS", "Числа и поиск сходства")
        arrow([(405, 143), (405, 97)])
        arrow([(365, 143), (365, 118), (239, 118), (239, 97)])
    elif kind == "chunks":
        scale = 450 / 1024
        for row, start in enumerate((0, 320, 640)):
            y = 135 - row * 48
            x = 12 + start * scale
            d.add(Rect(x, y, 384 * scale, 29, fillColor=PALE, strokeColor=FOREST, strokeWidth=.8))
            if row:
                d.add(Rect(x, y, 64 * scale, 29, fillColor=SAGE, strokeColor=None))
            text(x + 5, y + 10, f"{start} ... {start + 383}", 9.4)
        text(12, 9, "Шаг 320 токенов = длина 384 минус перекрытие 64", 10, MUTED)
    elif kind == "dense":
        node(0, 56, 142, 85, "Текст с префиксом", "query: вопрос\npassage: фрагмент")
        node(169, 56, 142, 85, "E5 на CPU", "384 нормализованных числа")
        node(338, 56, 142, 85, "FAISS IndexFlatIP", "Сходство с векторами базы")
        arrow([(142, 99), (167, 99)])
        arrow([(311, 99), (336, 99)])
        text(0, 18, "Результат поиска - ID фрагментов. Текст извлекается отдельно.", 10, MUTED)
    elif kind == "sign_links":
        node(0, 136, 210, 65, "Карточка знака 3.24", "gost-sign:3.24:1")
        node(278, 136, 202, 65, "Описание и общие условия", "rules_document_id")
        node(278, 31, 202, 65, "Применение по ГОСТ", "application_document_id")
        node(0, 31, 210, 65, "Изображение из каталога", "images/signs/3.24-1.png")
        arrow([(210, 169), (276, 169)])
        arrow([(105, 136), (105, 98)])
        arrow([(210, 151), (244, 151), (244, 65), (276, 65)])
        text(0, 6, "Связи хранятся в JSON. Отдельной графовой БД нет.", 9.8, MUTED)
    elif kind == "api_boundary":
        node(0, 72, 194, 105, "Приложение на ноутбуке", "Корпус, поиск, изображения\nПодготовка контекста")
        node(286, 72, 194, 105, "API-шлюз и генератор", "127.0.0.1:8080/v1\nМесто вычислений задаёт шлюз")
        arrow([(194, 136), (284, 136)])
        arrow([(286, 101), (196, 101)])
        text(235, 153, "запрос", 8.7, anchor="middle")
        text(235, 83, "ответ", 8.7, anchor="middle")
        text(0, 39, "Передаются вопрос, тексты источников и названия картинок.", 10)
        text(0, 21, "Пиксели и весь корпус приложение в этот API не отправляет.", 10, MUTED)
    elif kind in ("quality", "latency"):
        values = ([[.5429, .5613], [.6512, .4269]] if kind == "quality"
                  else [[.0337, 5.5493], [.0227, 8.4498]])
        maximum = 1 if kind == "quality" else 9
        x0, span = 100, 326
        for label, color, x in [("Hybrid", FOREST, 101), ("Hybrid + reranker", BLUE, 218)]:
            d.add(Rect(x, 163, 9, 9, fillColor=color, strokeColor=None))
            text(x + 15, 163, label, 9.5)
        for group, name in enumerate(("RuBQ", "SciFact")):
            y = 121 - group * 62
            text(0, y - 5, name, 11, font="Noto-Bold")
            for series, value in enumerate(values[group]):
                row_y = y - series * 22
                bar = value / maximum * span
                d.add(Rect(x0, row_y - 6, max(bar, 1), 14,
                           fillColor=(FOREST, BLUE)[series], strokeColor=None))
                value_label = f"{value:.4f}" if kind == "quality" else f"{value:.3f} с"
                text(x0 + bar + 6, row_y - 3, value_label.replace(".", ","), 9.3)
        text(x0, 5, "Одинаковые 50 запросов на каждом корпусе. Исторический тест.", 9, MUTED)
    elif kind == "atomic":
        node(0, 43, 142, 70, "Полная новая запись", "Временный файл")
        node(169, 43, 142, 70, "Атомарная замена", "os.replace + FileLock")
        node(338, 43, 142, 70, "Готовый JSON", "Читатель видит целый файл")
        arrow([(142, 78), (167, 78)])
        arrow([(311, 78), (336, 78)])
        text(0, 12, "При временной блокировке Windows выполняются ограниченные повторы записи.", 9.5, MUTED)
    return d

FIGURES = {
    "rag": "Принцип RAG: поиск материала перед подготовкой ответа",
    "architecture": "Архитектура личного RAG по ПДД и основные модули",
    "indexing": "Подготовка поисковых индексов до вопросов пользователя",
    "chunks": "Окна токенов с перекрытием. Позиции приведены для объяснения алгоритма",
    "dense": "Смысловой поиск: E5 превращает текст в числа, FAISS находит сходство",
    "sign_links": "Связи карточки знака с правилами и изображением",
    "api_boundary": "Разделение локального поиска и API генерации",
    "quality": "nDCG@10 в историческом эксперименте Laya. Больше - лучше",
    "latency": "p50 времени поиска с reranker в историческом эксперименте. Меньше - быстрее",
    "atomic": "Сохранение JSON без промежуточного обрезанного состояния",
}

class ReportDoc(BaseDocTemplate):
    def __init__(self, path, **kwargs):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=58, bottomMargin=55, title="Личный RAG по ПДД России: подробный технический отчёт",
                         author="epitaph76", **kwargs)
        frame = Frame(MARGIN, 55, WIDTH, PAGE_H - 113, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=self.decorate))

    def beforeDocument(self):
        self.heading_id = 0

    def decorate(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, PAGE_H - 33, PAGE_W - MARGIN, PAGE_H - 33)
        canvas.setFont("Noto", 8.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H - 25, "Личный RAG по ПДД России")
        canvas.drawRightString(PAGE_W - MARGIN, 31, str(doc.page))
        canvas.drawString(MARGIN, 31, "Срез проекта: 31.10.2026")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph) or flowable.style.name not in ("Heading1", "Heading2"):
            return
        level = 0 if flowable.style.name == "Heading1" else 1
        title = flowable.getPlainText()
        key = f"heading-{self.heading_id}"
        self.heading_id += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=False)
        if level == 0:
            self.notify("TOCEntry", (0, title, self.page, key))

def make_table(lines, s):
    rows = [[part.strip() for part in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
    count = len(rows[0])
    weights = {2: [.32, .68], 3: [.26, .25, .49], 4: [.31, .23, .23, .23],
               5: [.24, .19, .19, .19, .19], 6: [.26, .16, .145, .145, .145, .145]}.get(count, [1 / count] * count)
    data = [[Paragraph(inline(cell), s["thead" if i == 0 else "cell"]) for cell in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=[WIDTH * w for w in weights], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), FOREST), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f3")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), .6, FOREST), ("LINEBELOW", (0, 1), (-1, -1), .4, LINE),
    ]))
    return table

def compile_markdown(markdown, s):
    story, output_md = [], []
    lines = markdown.splitlines()
    i, figures = 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            if story:
                story.append(PageBreak())
            story.append(Paragraph(inline(line[2:]), s["h1"]))
            output_md.append(line + "\n")
            i += 1
        elif line.startswith(("## ", "### ")):
            prefix = "### " if line.startswith("### ") else "## "
            story.append(Paragraph(inline(line[len(prefix):]), s["h3" if prefix == "### " else "h2"]))
            output_md.append(line + "\n")
            i += 1
        elif line.startswith("{{figure:"):
            kind = line.removeprefix("{{figure:").removesuffix("}}")
            drawing = diagram(kind)
            figures += 1
            drawing.hAlign = "CENTER"
            renderSVG.drawToFile(drawing, str(ASSETS / (kind + ".svg")))
            caption = f"Рисунок {figures}. {FIGURES[kind]}"
            story.append(KeepTogether([Spacer(1, 6), drawing, Paragraph(inline(caption), s["caption"])]))
            output_md.append(f"![{caption}](report_assets/{kind}.svg)\n\n{caption}\n")
            i += 1
        elif line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            caption, relative = match.groups()
            path = ROOT / "docs" / relative
            image = Image(str(path))
            factor = min(WIDTH / image.imageWidth, 300 / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * factor, image.imageHeight * factor
            image.hAlign = "CENTER"
            figures += 1
            elements = [Spacer(1, 5), image, Paragraph(inline(f"Рисунок {figures}. {caption}"), s["caption"])]
            block = KeepTogether(elements)
            block.report_followup = elements
            story.append(block)
            output_md.append(line + "\n")
            i += 1
        elif line.startswith("```"):
            content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                content.append(lines[i])
                i += 1
            text = "<br/>".join(escape(value).replace(" ", "&#160;") for value in content)
            story.append(Paragraph(text, s["code"]))
            output_md.append("```text\n" + "\n".join(content) + "\n```\n")
            i += 1
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.extend([make_table(table_lines, s), Spacer(1, 11)])
            output_md.append("\n".join(table_lines) + "\n")
        elif line.startswith("- "):
            story.append(Paragraph("- " + inline(line[2:]), s["bullet"]))
            output_md.append(line + "\n")
            i += 1
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![", "{{", "```", "- ")):
                paragraph.append(lines[i].strip())
                i += 1
            text = " ".join(paragraph)
            reference = text.startswith(("Реализация:", "Источники:", "Источники реализации:",
                "Первичный источник понятия:", "Основные материалы проекта:", "Код выбора источников", "Доказательства:"))
            flowable = Paragraph(inline(text), s["refs" if reference else "body"])
            if story and hasattr(story[-1], "report_followup"):
                previous = story.pop()
                story.append(KeepTogether(previous.report_followup + [flowable]))
            elif reference and story and isinstance(story[-1], Paragraph):
                previous = story.pop()
                story.append(KeepTogether([previous, flowable]))
            else:
                story.append(flowable)
            output_md.append(text + "\n")
    return story, "\n".join(output_md), figures

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/PDD_RAG_Detailed_Report_RU.pdf")
    args = parser.parse_args()
    register_fonts()
    s = styles()
    facts = json.loads((ASSETS / "facts.json").read_text("utf-8"))
    sources = sorted(SOURCE.glob("*.md"))
    if not sources:
        raise ValueError("No report chapters")
    markdown = "\n\n".join(path.read_text("utf-8-sig") for path in sources)
    body, public_markdown, figures = compile_markdown(markdown, s)
    cover = [Spacer(1, 57),
        Paragraph("ПРОЕКТНЫЙ ОТЧЁТ", ParagraphStyle("Kicker", fontName="JetBrains", fontSize=10, leading=15, textColor=FOREST)),
        Spacer(1, 23),
        Paragraph("Личный RAG<br/>по ПДД России", ParagraphStyle("Cover", fontName="Noto-Bold", fontSize=35, leading=43, textColor=FOREST)),
        Spacer(1, 23),
        Paragraph("Как устроен помощник, который ищет правила,<br/>показывает дорожные знаки и объясняет источники", ParagraphStyle("CoverSub", fontName="Noto", fontSize=15, leading=23, textColor=INK)),
        Spacer(1, 37),
        Paragraph("Подробное объяснение для читателя без опыта в RAG.<br/>Данные, хранение, алгоритмы, модели, архитектура и результаты проверок.", s["body"]),
        Spacer(1, 111),
        Paragraph("31 октября 2026 года", s["body"]),
        Paragraph(f"Репозиторий: <a href='https://github.com/epitaph76/RAG' color='#3f553e'>github.com/epitaph76/RAG</a><br/>Срез реализации: {facts['source_commit'][:7]}<br/>Снимок нормативного корпуса: 30.10.2026", s["caption"]),
        PageBreak(),
        Paragraph("Оглавление", ParagraphStyle("ContentsTitle", parent=s["h1"])),
        Paragraph("Главы идут от основных понятий к реализации. Термины собраны также в приложении Б. В PDF доступны закладки и ссылки.", s["body"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [s["toc"]]
    toc.dotsMinLevel = 0
    cover.extend([Spacer(1, 15), toc, PageBreak()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDoc(args.output)
    rl_config.warnOnMissingFontGlyphs = 1
    doc.multiBuild(cover + body)
    md_path = ROOT / "docs/PDD_RAG_DETAILED_REPORT_RU.md"
    md_path.write_text("# Личный RAG по ПДД России\n\nПодробный технический отчёт. 31.10.2026. Срез реализации " + facts['source_commit'][:7] + ".\n\n" + public_markdown, encoding="utf-8")
    print(json.dumps({"pdf": str(args.output), "markdown": str(md_path), "figures": figures,
                      "words_approx": len(markdown.split()), "source_chapters": len(sources)}, ensure_ascii=False))

if __name__ == "__main__":
    main()

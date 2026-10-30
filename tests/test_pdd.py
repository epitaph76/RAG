"""Parser regressions for legal footnotes, table alignment and provenance."""
from bs4 import BeautifulSoup
import pytest

from raglab.pdd import clean, edition, expand_table, legal_blocks, split_blocks, sign_definitions
from raglab.retrieval import prioritize_sign_codes
from raglab.types import Chunk, Hit


def test_legal_parser_keeps_operative_footnote_and_table_but_removes_publisher_notes():
    soup = BeautifulSoup('''<div class="document-page__content"><h1>ПДД</h1>
        <p><a id="dst1"></a>2.1.1. Передавать документы для проверки &lt;*&gt;:</p>
        <div class="doc-edit"><p>Редакционная история</p></div>
        <p>&lt;*&gt; Электронное предъявление приравнивается к предъявлению документов.</p>
        <div class="doc-insert"><p>КонсультантПлюс: примечание</p></div>
        <p>водительское удостоверение;</p>
        <table><tr><td><p>Допустимое значение</p></td><td>0,16</td></tr></table>
        <p><a id="dst2"></a>2.1.2. Пристегиваться.</p></div>''', 'html.parser')
    blocks = legal_blocks(soup)
    groups = split_blocks(blocks)
    assert len(groups) == 2
    assert groups[0][1] == "dst1"
    text = " ".join(groups[0][0])
    assert "Электронное предъявление" in text
    assert "водительское удостоверение" in text
    assert text.count("Допустимое значение") == 1
    assert "0,16" in text
    assert "Редакционная история" not in text and "КонсультантПлюс" not in text


def test_standard_table_rowspan_keeps_shared_name_aligned_with_each_code():
    table = BeautifulSoup('''<table>
        <tr><td>image A</td><td rowspan="2">Общее название</td><td colspan="2">image B</td><td rowspan="2">Иное название</td></tr>
        <tr><td>1.1</td><td>1.4.1</td><td>1.4.2</td></tr>
        </table>''', 'html.parser').table
    grid = expand_table(table)
    assert clean(grid[1, 1].get_text()) == "Общее название"
    assert clean(grid[1, 2].get_text()) == "1.4.1"
    assert clean(grid[1, 3].get_text()) == "1.4.2"
    assert grid[0, 2] is grid[0, 3]
    assert clean(grid[1, 4].get_text()) == "Иное название"


def test_standard_clauses_without_trailing_period_remain_separate():
    groups = split_blocks([("5.6.44 Знак 5.41 применяют...", "a"),
        ("Дополнительное условие.", ""), ("5.6.45 Знак 5.42 применяют...", "b")], "standard")
    assert len(groups) == 2
    assert len(groups[0][0]) == 2
    assert groups[1][1] == "b"


def test_captcha_is_not_accepted_as_legal_text():
    with pytest.raises(ValueError, match="Missing legal content"):
        legal_blocks(BeautifulSoup("<h1>Подтвердите, что вы не робот</h1>", "html.parser"))


def test_edition_fallback_uses_latest_dated_amending_act():
    soup = BeautifulSoup('<div class="doc-roll__content">от 04.08.2026 N 330-ФЗ, от 07.04.2026 N 21-П</div>', 'html.parser')
    assert edition(soup) == "2026-08-04"


def test_explicit_sign_identifier_recovers_catalog_even_when_dense_misses_it():
    correct = Chunk("gost-sign:6.2.1:1:0", "gost-sign:6.2.1:1", "Рекомендуемая скорость")
    wrong = Hit(Chunk("other:0", "gost-sign:6.22:1", "Фотовидеофиксация"), .9)
    hits = prioritize_sign_codes("Что означает знак 6.2.1?", [wrong], [wrong.chunk, correct])
    assert hits[0].chunk is correct
    assert len(hits) == 2
    assert prioritize_sign_codes("Что означает знак 6.2.10?", [wrong], [correct]) == [wrong]
    assert prioritize_sign_codes("Объясни пункт 6.2.1", [wrong], [correct]) == [wrong]


def test_exception_with_a_quoted_law_title_does_not_replace_sign_definition():
    from types import SimpleNamespace
    html = '''<div class="document-page__content"><h1>Запрещающие знаки</h1>
        <p>3.28 "Стоянка запрещена". Запрещается стоянка.</p>
        <p>3.29 "Стоянка запрещена по нечетным числам месяца".</p>
        <p>3.28 - 3.30 - на транспортные средства инвалидов по Федеральному закону "О социальной защите инвалидов";</p>
        </div>'''
    collector = SimpleNamespace(soup=lambda _: BeautifulSoup(html, "html.parser"))
    definitions = sign_definitions(collector, [("sign_rules", "3", "Знаки", "https://example.com", "2026-08-28")])
    assert definitions["3.28"]["text"].startswith('3.28 "Стоянка запрещена"')
    assert "Запрещается стоянка" in definitions["3.28"]["text"]

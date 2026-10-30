import hashlib
import json
from pathlib import Path
import pytest

from raglab.traffic import TrafficKnowledge, mentions_code, requested_codes, relevant_rules
from raglab.types import Document, Chunk, Hit

@pytest.fixture
def knowledge(tmp_path):
    image = b"verified-image-fixture"
    (tmp_path / "images").mkdir()
    (tmp_path / "images/sign.png").write_bytes(image)
    rows = [{"id": f"gost-sign:{code}:1", "code": code, "name": name,
             "local_path": "images/sign.png", "sha256": hashlib.sha256(image).hexdigest(),
             "kind": "reference_illustration", "rules_document_id": "rules", "application_document_id": f"use:{code}"}
            for code, name in [("3.27", "Остановка запрещена"), ("3.28", "Стоянка запрещена")]]
    (tmp_path / "signs.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (tmp_path / "photos.jsonl").write_text(json.dumps({**rows[0], "id": "photo:1", "kind": "real_world_photo_crop"}), encoding="utf-8")
    docs = [Document(row["id"], f"Знак {row['code']}. Определение.\n\nГОСТ Р 52289-2019 · установка", row["name"]) for row in rows]
    docs += [Document("rules", 'Знаки\n\nДействие знаков не распространяется:\n\n3.27 - на маршрутные транспортные средства;\n\n3.28 - 3.30 - исключение для стоянки.\n\nЗона действия знаков 3.27 - 3.30 до перекрестка.\n\nДействие 3.27 - 3.30 только на сторону дороги.'),
             Document("use:3.27", "Установка 3.27", url="https://example.com/standard"),
             Document("use:3.28", "Установка 3.28")]
    return TrafficKnowledge(docs, tmp_path)

def test_sign_names_bare_codes_and_article_numbers_are_distinct(knowledge):
    assert requested_codes("3.27?", knowledge.signs) == ["3.27"]
    assert requested_codes('Что значит «Остановка запрещена»?', knowledge.signs) == ["3.27"]
    assert requested_codes("Объясни пункт 3.27", knowledge.signs) == []
    assert requested_codes("Сравни знаки 3.27 и 3.28", knowledge.signs) == ["3.27", "3.28"]
    assert requested_codes("Покажи знаки 3.27 и 3.28. Чем они отличаются?", knowledge.signs) == ["3.27", "3.28"]
    assert requested_codes("3.27.", knowledge.signs) == ["3.27"]

def test_rule_ranges_do_not_confuse_codes_or_prefixes():
    assert mentions_code("3.27 - 3.30", "3.28")
    assert not mentions_code("3.27 - 3.30", "3.24")
    assert not mentions_code("3.24", "3.2")
    assert not mentions_code("3.18.1 - 3.18.2", "3.18")
    assert mentions_code("3.27 - 3.30.", "3.28")

def test_sign_context_recovers_definition_exceptions_zone_and_standard(knowledge):
    wrong = Hit(Chunk("other:0", "other", "Нерелевантный фрагмент"), .9)
    sources = knowledge.sources("Объясни знак 3.27", [wrong])
    assert [hit.chunk.document_id for hit in sources] == ["gost-sign:3.27:1", "rules", "use:3.27", "other"]
    assert "ГОСТ Р 52289" not in sources[0].chunk.text
    rules = sources[1].chunk.text
    assert "не распространяется" in rules and "маршрутные" in rules
    assert "Зона действия" in rules and "сторону дороги" in rules
    assert "исключение для стоянки" not in rules
    assert sources[2].chunk.url == "https://example.com/standard"

def test_two_signs_have_distinct_images_and_shared_rules_once(knowledge):
    sources = knowledge.sources("Сравни знаки 3.27 и 3.28", [])
    assert [hit.chunk.document_id for hit in sources[:3]] == ["gost-sign:3.27:1", "gost-sign:3.28:1", "rules"]
    images = knowledge.images("Сравни знаки 3.27 и 3.28", sources, "Объяснение [1][2][3].")
    assert [image["code"] for image in images] == ["3.27", "3.28"]
    assert [image["source_number"] for image in images] == [1, 2]

def test_unknown_sign_does_not_show_a_different_sign(knowledge):
    wrong = Hit(Chunk("sign:0", "gost-sign:3.27:1", "Знак 3.27"), 1)
    assert knowledge.sources("Покажи знак 9.99", [wrong]) == []
    assert knowledge.images("Общие правила", [wrong], "Ответ без ссылок") == []

def test_photos_are_opt_in_and_linked_to_the_same_code(knowledge):
    sources = knowledge.sources("Знак 3.27", [])
    assert len(knowledge.images("Знак 3.27", sources, "Ответ [1]")) == 1
    images = knowledge.images("Покажи фото знака 3.27", sources, "Ответ [1]")
    assert len(images) == 2 and {row["code"] for row in images} == {"3.27"}

def test_only_verified_catalog_assets_can_be_displayed(knowledge):
    asset = knowledge.signs[0]
    assert knowledge.image_bytes(asset) == b"verified-image-fixture"
    assert knowledge.image_bytes({**asset, "local_path": "../../secret"}) == b"verified-image-fixture"
    (knowledge.directory / "images/sign.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="целостности"):
        knowledge.image_bytes(asset)
    with pytest.raises(ValueError, match="каталоге"):
        knowledge.image_bytes({"id": "unknown"})
    knowledge.signs[0]["local_path"] = "../outside.png"
    with pytest.raises(ValueError, match="путь"):
        knowledge.image_bytes(asset)

def test_expired_paragraph_is_not_reintroduced_as_a_rule():
    text = "Знаки\n\n3.27 Правило.\n\n3.27 - абзац утратил силу."
    assert "утратил" not in relevant_rules(text, ["3.27"])
    text = "Знаки\n\nДействие не распространяется:\n\n3.27 - исключение А.\n\n3.28 - исключение Б."
    selected = relevant_rules(text, ["3.28"])
    assert "не распространяется" in selected and "исключение Б" in selected and "исключение А" not in selected

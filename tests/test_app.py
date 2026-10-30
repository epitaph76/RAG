from pathlib import Path
import base64
import hashlib
from types import SimpleNamespace
import pytest
from streamlit.testing.v1 import AppTest
import raglab.config as config
import raglab.data as data
import raglab.retrieval as retrieval
import raglab.generation as generation
from raglab.storage import write_json
from raglab.types import Document, Chunk, Hit, SearchResult, Answer

APP = str(Path(__file__).resolve().parents[1] / "app.py")

@pytest.fixture
def service(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    calls = []
    source = Hit(Chunk("doc:0", "doc", "ПДД России: правила движения.", "ПДД РФ", "https://example.org/pdd"), 1)
    def dataset(name, *args, **kwargs):
        return SimpleNamespace(name=name, documents=[Document("gost-sign:3.24:1", "Знак 3.24: ограничение максимальной скорости.", "Знак 3.24")], manifest={"documents": 1})
    def index(ds, models):
        return SimpleNamespace(dataset=ds, root=tmp_path / "artifacts" / "indices" / ds.name,
            bm25=True, settings=SimpleNamespace(context_k=4),
            retrieve=lambda question, mode: SearchResult([source], "hybrid", None, {"total": 1}))
    def answer(question, sources, profile, api_key=None, **kwargs):
        calls.append((question, profile.name))
        return Answer("Ответ по ПДД [1].", [1], [], False, {"total_tokens": 30}, 10, profile.model, profile.name, sources)
    monkeypatch.setattr(data, "prepare", dataset)
    monkeypatch.setattr(retrieval, "Index", index)
    monkeypatch.setattr(generation, "generate", answer)
    for name in ("pdd",):
        write_json(tmp_path / "data" / name / "manifest.json", {"documents": 1})
        write_json(tmp_path / "artifacts" / "indices" / name / "manifest.json", {})
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=")
    image_path = tmp_path / "data/pdd/images/sign.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image)
    import json
    (image_path.parents[1] / "signs.jsonl").write_text(json.dumps({"id": "gost-sign:3.24:1", "code": "3.24",
        "name": "Ограничение максимальной скорости", "kind": "reference_illustration", "standard": "ГОСТ",
        "local_path": "images/sign.png", "sha256": hashlib.sha256(image).hexdigest(), "source_url": "https://example.org/sign"}), encoding="utf-8")
    return calls, tmp_path

def button(app, label):
    return next(item for item in app.button if item.label == label)

def selectbox(app, label):
    return next(item for item in app.selectbox if item.label == label)

def test_main_screen_hides_settings_and_generates_only_on_submit(service):
    calls, _ = service
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert len(app.selectbox) == 0 and len(app.tabs) == 0
    assert not app.sidebar
    assert not app.chat_input[0].disabled
    assert calls == []
    assert not any("С чего начнём" in text.value for text in app.markdown)
    assert app.session_state["preferences"]["dataset"] == "pdd"
    app.chat_input[0].set_value("Где запрещена остановка?").run()
    assert not app.exception
    assert len(calls) == 1 and len(app.chat_message) == 2
    app.run()
    assert len(calls) == 1
    assert not any(expander.label == "Диагностика ответа" for expander in app.expander)
    button(app, "Новый чат").click().run()
    assert not app.chat_message and len(calls) == 1

def test_settings_survive_closing_without_generation_or_key_mixing(service):
    calls, _ = service
    app = AppTest.from_file(APP, default_timeout=20).run()
    button(app, "Настройки").click().run()
    assert not app.exception
    assert not any(item.label == "База знаний" for item in app.selectbox)
    assert [item.label for item in app.tabs] == ["Поиск", "Подключение", "О базе"]
    selectbox(app, "Способ поиска").select("full").run()
    next(item for item in app.text_input if item.label == "Ключ API").set_value("fixture-sub2api-key").run()
    selectbox(app, "Сервис ответов").select("routerai").run()
    assert next(item for item in app.text_input if item.label == "Ключ API").value == ""
    next(item for item in app.text_input if item.label == "Ключ API").set_value("fixture-router-key").run()
    selectbox(app, "Сервис ответов").select("sub2api").run()
    assert next(item for item in app.text_input if item.label == "Ключ API").value == "fixture-sub2api-key"
    button(app, "Готово").click().run()
    assert not app.exception
    assert len(app.selectbox) == 0 and calls == []
    assert app.session_state["preferences"]["mode"] == "full"
    button(app, "Настройки").click().run()
    assert selectbox(app, "Способ поиска").value == "full"
    assert app.session_state["api_keys"]["routerai"] == "fixture-router-key"
    assert calls == []

def test_sign_image_is_in_the_answer_and_rerenders_without_a_new_request(service):
    calls, _ = service
    app = AppTest.from_file(APP, default_timeout=20).run()
    app.chat_input[0].set_value("Покажи знак 3.24").run()
    assert not app.exception and len(calls) == 1
    assert len(app.get("image")) == 1
    app.run()
    assert len(app.get("image")) == 1 and len(calls) == 1
    button(app, "Новый чат").click().run()
    assert not app.get("image") and len(calls) == 1

def test_background_benchmark_disables_duplicate_heavy_jobs(service):
    import os
    import psutil
    _, tmp_path = service
    write_json(tmp_path / "artifacts" / "benchmark-running.json", {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    app = AppTest.from_file(APP, default_timeout=20).run()
    button(app, "Настройки").click().run()
    assert not app.exception
    for label in ("Подготовить базу", "Перестроить индекс"):
        assert button(app, label).disabled

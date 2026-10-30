from pathlib import Path
from types import SimpleNamespace
import pytest
from streamlit.testing.v1 import AppTest
import raglab.config as config
import raglab.data as data
import raglab.retrieval as retrieval
import raglab.generation as generation
from raglab.storage import write_json
from raglab.types import Chunk, Hit, SearchResult, Answer

APP = str(Path(__file__).resolve().parents[1] / "app.py")

@pytest.fixture
def service(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    calls = []
    source = Hit(Chunk("doc:0", "doc", "Instagram появился 6 октября 2010 года.", "Wikipedia · 1", "https://ru.wikipedia.org/?curid=1"), 1)
    def dataset(name, *args, **kwargs):
        return SimpleNamespace(name=name, manifest={"documents": 1})
    def index(ds, models):
        return SimpleNamespace(dataset=ds, root=tmp_path / "artifacts" / "indices" / ds.name,
            bm25=True, settings=SimpleNamespace(context_k=4),
            retrieve=lambda question, mode: SearchResult([source], "hybrid", None, {"total": 1}))
    def answer(question, sources, profile, api_key=None):
        calls.append((question, profile.name))
        return Answer("Instagram создан в 2010 году [1].", [1], [], False, {"total_tokens": 30}, 10, profile.model, profile.name, sources)
    monkeypatch.setattr(data, "prepare", dataset)
    monkeypatch.setattr(retrieval, "Index", index)
    monkeypatch.setattr(generation, "generate", answer)
    for name in ("rubq", "scifact"):
        write_json(tmp_path / "data" / name / "manifest.json", {"documents": 1})
        write_json(tmp_path / "artifacts" / "indices" / name / "manifest.json", {})
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
    app.chat_input[0].set_value("В каком году появился Instagram?").run()
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

def test_background_benchmark_disables_duplicate_heavy_jobs(service):
    import os
    import psutil
    _, tmp_path = service
    write_json(tmp_path / "artifacts" / "benchmark-running.json", {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    app = AppTest.from_file(APP, default_timeout=20).run()
    button(app, "Настройки").click().run()
    assert not app.exception
    for label in ("Подготовить базу", "Перестроить индекс", "Запустить эксперимент"):
        assert button(app, label).disabled

from pathlib import Path
from streamlit.testing.v1 import AppTest
import raglab.config as config
import raglab.generation as generation

def test_empty_app_and_rerenders_do_not_generate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected API request")
    monkeypatch.setattr(generation, "generate", forbidden)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Чат с источниками", "Эксперименты"]
    next(b for b in app.button if b.label == "Найти и ответить").click().run()
    assert not app.exception
    assert any("подготов" in w.value.lower() for w in app.warning)
    next(s for s in app.selectbox if s.label == "API-профиль").select("routerai").run()
    next(s for s in app.selectbox if s.label == "Конфигурация").select("full").run()
    assert not app.exception
    assert any("deepseek" in c.value for c in app.caption)

def test_background_benchmark_disables_duplicate_heavy_jobs(monkeypatch, tmp_path):
    import os
    import psutil
    from raglab.storage import write_json
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    write_json(tmp_path / "artifacts" / "benchmark-running.json",
               {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    for label in ("Подготовить данные и индекс", "Перестроить индекс", "Запустить эксперимент"):
        assert next(button for button in app.button if button.label == label).disabled

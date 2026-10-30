import os
from dataclasses import asdict
import pandas as pd
import streamlit as st

from raglab.config import DATA, ARTIFACTS, LABELS, Settings, profiles
from raglab.data import prepare
from raglab.models import Models
from raglab.retrieval import Index
from raglab.generation import Budget, generate, GenerationError
from raglab.evaluation import evaluate, evaluate_qa
from raglab.storage import read_json

st.set_page_config(page_title="RAG · Laya Lab", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background: #f6f7f4;} h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stSidebar"] {background:#e8eee8;}
[data-testid="stMetric"] {background:white;padding:16px;border-radius:12px;border:1px solid #dce4dc;}
[data-testid="stChatMessage"] {background:white;border:1px solid #e2e7e1;border-radius:14px;}
.lab-kicker {font:600 12px monospace;letter-spacing:2px;color:#47735c;margin-bottom:8px;}
</style>""", unsafe_allow_html=True)

def benchmark_is_running():
    import psutil
    path = ARTIFACTS / "benchmark-running.json"
    try:
        state = read_json(path)
        process = psutil.Process(state["pid"])
        return abs(process.create_time() - state["create_time"]) < 0.01 and process.is_running()
    except (OSError, ValueError, KeyError, psutil.Error):
        return False

benchmark_busy = benchmark_is_running()

@st.cache_data(show_spinner=False)
def embedding_total(order_path, modified_ns):
    return len(read_json(order_path))

@st.fragment(run_every="10s")
def background_progress():
    if not benchmark_is_running():
        st.success("Фоновый benchmark завершён. Обновите страницу, чтобы загрузить готовые индексы и отчёты.")
        return
    for name in ("rubq", "scifact"):
        directory = ARTIFACTS / "indices" / name
        pending = directory / "pending.json"
        order = directory / "embedding-order.json"
        if pending.exists() and order.exists() and not (directory / "manifest.json").exists():
            state = read_json(pending)
            total = embedding_total(order, order.stat().st_mtime_ns)
            done = min(state["completed"], total)
            label = "RuBQ" if name == "rubq" else "SciFact"
            st.progress(done / total if total else 0, text=f"{label} · проиндексировано {done:,} из {total:,} чанков · осталось {total - done:,}")
            st.caption("Локальный расчёт на CPU. Счётчик обновляется каждые 10 секунд; API-запросы не расходуются.")
            return
    st.info("Индексы готовы. Выполняется локальное сравнение retrieval и Laya; результаты сохраняются в отчёты.")

@st.cache_resource
def model_resources():
    return Models()

@st.cache_resource
def index_resource(name, manifest_mtime):
    return Index(prepare(name), model_resources())

def show_sources(sources):
    for number, hit in enumerate(sources, 1):
        chunk = hit["chunk"] if isinstance(hit, dict) else asdict(hit.chunk)
        score = hit["score"] if isinstance(hit, dict) else hit.score
        with st.expander(f"[{number}] {chunk['title'] or chunk['document_id']} · {score:.4f}"):
            st.write(chunk["text"])
            st.caption(f"Документ {chunk['document_id']} · чанк {chunk['id']}")
            if chunk.get("url"):
                st.link_button("Открыть источник ↗", chunk["url"])

with st.sidebar:
    st.markdown("### ◈ Laya Lab")
    st.caption("HYBRID RETRIEVAL WORKSPACE")
    dataset_name = st.selectbox("Датасет", ["rubq", "scifact"], format_func=lambda x: "RuBQ 2.0 · русский" if x == "rubq" else "SciFact · English")
    mode = st.selectbox("Конфигурация", list(LABELS), index=2, format_func=LABELS.get)
    st.divider()
    profile_name = st.selectbox("API-профиль", list(profiles()))
    profile = profiles()[profile_name]
    st.caption(f"{profile.model}\n\n{profile.base_url}")
    key = st.text_input("API-ключ для этой сессии", type="password", key=f"key_{profile_name}",
                        help=f"Можно оставить пустым, если задана переменная {profile.key_env}. Ключ не сохраняется на диск.")
    state = Budget().status()
    st.caption(f"API-попытки: {state['used']} / {state['limit']} · повторы отключены")
    st.divider()
    build_clicked = st.button("Подготовить данные и индекс", width="stretch", disabled=benchmark_busy)
    rebuild_clicked = st.button("Перестроить индекс", width="stretch", disabled=benchmark_busy)
    if benchmark_busy:
        st.info("Выполняется фоновый benchmark. Подготовка индексов временно отключена, чтобы не запускать второй тяжёлый процесс.")
    st.caption("CPU · E5 small · Laya multilingual\n\nПервый запуск загружает модели. Индексы сохраняются на диске.")

st.markdown('<div class="lab-kicker">LOCAL RESEARCH / RU + EN</div>', unsafe_allow_html=True)
st.title("Поиск, которому есть на что сослаться.")
st.caption("BM25 + Dense + RRF → Laya → ответ с источниками")
if benchmark_busy:
    background_progress()

index = None
manifest_path = DATA / dataset_name / "manifest.json"
if build_clicked or rebuild_clicked:
    bar = st.progress(0, text="Подготовка данных")
    try:
        dataset = prepare(dataset_name, lambda p, m: bar.progress(p, text=m))
        index = index_resource(dataset_name, manifest_path.stat().st_mtime_ns)
        (index.rebuild if rebuild_clicked else index.build)(lambda p, m: bar.progress(p, text=m))
        st.success("Индекс готов. Можно задавать вопросы.")
    except Exception as error:
        st.error(f"Не удалось подготовить индекс ({type(error).__name__}). Подробности доступны при запуске CLI; проверьте сеть и свободную память.")
    bar.empty()
elif manifest_path.exists():
    try:
        index = index_resource(dataset_name, manifest_path.stat().st_mtime_ns)
        if index.bm25 is None and (index.root / "manifest.json").exists():
            with st.spinner("Загрузка сохранённого индекса…"):
                index.build()
    except Exception as error:
        st.error(f"Не удалось загрузить индекс ({type(error).__name__}). Нажмите «Перестроить индекс».")

cols = st.columns(3)
cols[0].metric("Документов", f"{len(index.dataset.documents):,}" if index else "—")
cols[1].metric("Чанков в индексе", f"{len(index.chunks):,}" if index and index.bm25 is not None else "—")
cols[2].metric("Генератор", "Luna" if profile_name == "sub2api" else "Flash", help=profile.model)
chat_tab, experiment_tab = st.tabs(["Чат с источниками", "Эксперименты"])

with chat_tab:
    st.caption("Вопросы независимы: история диалога не отправляется генератору.")
    histories = st.session_state.setdefault("histories", {})
    history = histories.setdefault(dataset_name, [])
    if st.button("Очистить историю"):
        histories[dataset_name] = []
        history = histories[dataset_name]
    for record in history:
        with st.chat_message("user"):
            st.write(record["question"])
        with st.chat_message("assistant"):
            st.markdown(record["text"])
            st.caption(record["caption"])
            for warning in record.get("warnings", []):
                st.warning(warning)
            show_sources(record["sources"])
            with st.expander("Задержки и токены"):
                st.json(record["details"])
    with st.form("question_form", clear_on_submit=False):
        question = st.text_area("Ваш вопрос", placeholder="Например: Кто написал роман «Хижина дяди Тома»?", max_chars=4000)
        use_api = st.checkbox("Сгенерировать ответ через API", value=True)
        submitted = st.form_submit_button("Найти и ответить", type="primary")
    if submitted:
        if not index or index.bm25 is None:
            st.warning("Сначала нажмите «Подготовить данные и индекс».")
        elif not question.strip():
            st.warning("Введите вопрос.")
        else:
            try:
                with st.spinner("Ищу источники и готовлю ответ…"):
                    result = index.retrieve(question, mode)
                    sources = result.hits[:index.settings.context_k]
                    warnings = result.warnings[:]
                    answer = generate(question, sources, profile, key or None) if use_api else None
                    if answer:
                        if answer.invalid_citations:
                            warnings.append(f"Неверные номера источников: {answer.invalid_citations}")
                        if answer.missing_citations:
                            warnings.append("Ответ не содержит ссылок на переданные источники.")
                    history.append({"question": question, "text": answer.text if answer else ("Найденные фрагменты готовы для просмотра." if sources else "По запросу не найдено фрагментов."),
                        "caption": f"{LABELS[mode]} · маршрут: {result.route} · поиск: {result.timings['total'] / 1000:.2f} с",
                        "sources": [asdict(h) for h in sources], "warnings": warnings,
                        "details": {"timings_ms": result.timings, "router_confidence": result.confidence,
                                    "generation_ms": answer.latency_ms if answer else None, "usage": answer.usage if answer else {}}})
                st.rerun()
            except GenerationError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Поиск не выполнен ({type(error).__name__}). Проверьте индекс и повторите запрос.")

with experiment_tab:
    st.subheader("Сравнение на одинаковых запросах")
    st.write("Baseline оценивается на полной тестовой выборке. Laya — на фиксированных 50 размеченных вопросах; Hybrid повторяется на той же выборке.")
    if index:
        st.json(index.dataset.manifest["splits"], expanded=False)
    suite = st.radio("Набор экспериментов", ["Baseline · полный test", "Laya · 50 вопросов", "QA · 18 API-попыток"], horizontal=True)
    if st.button("Запустить эксперимент", type="primary", disabled=benchmark_busy):
        if not index or index.bm25 is None:
            st.warning("Сначала подготовьте индекс.")
        elif suite.startswith("QA") and dataset_name != "rubq":
            st.warning("QA с эталонными ответами доступен для RuBQ.")
        else:
            bar = st.progress(0, text="Запуск эксперимента")
            try:
                callback = lambda p, m: bar.progress(min(p, 1), text=m)
                if suite.startswith("QA"):
                    path, report = evaluate_qa(index, profile, key or None, callback)
                else:
                    modes = ("bm25", "dense", "hybrid") if suite.startswith("Baseline") else ("hybrid", "router", "reranker", "full")
                    path, report = evaluate(index, modes, None if suite.startswith("Baseline") else 50, callback)
                st.success(f"Отчёт сохранён: {path.name}")
            except GenerationError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Эксперимент остановлен ({type(error).__name__}). Индексы и завершённые QA-ответы сохранены.")
            bar.empty()
    reports = sorted((ARTIFACTS / "reports").glob("*.json"), reverse=True)
    if reports:
        selected = st.selectbox("Сохранённый отчёт", reports, format_func=lambda p: p.stem)
        report = read_json(selected)
        st.dataframe(pd.DataFrame(report.get("summary", [])), hide_index=True, width="stretch")
        if report.get("note"):
            st.caption(report["note"])
        c1, c2 = st.columns(2)
        c1.download_button("Скачать JSON", selected.read_bytes(), selected.name, "application/json")
        if selected.with_suffix(".csv").exists():
            c2.download_button("Скачать CSV", selected.with_suffix(".csv").read_bytes(), selected.with_suffix(".csv").name, "text/csv")
    else:
        st.info("После первого эксперимента здесь появятся метрики и отчёты для скачивания.")

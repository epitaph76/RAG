from dataclasses import asdict
from html import escape
import mimetypes
from urllib.parse import urlparse
import pandas as pd
import streamlit as st

from raglab.config import ROOT, DATA, ARTIFACTS, LABELS, profiles
from raglab.data import prepare
from raglab.models import Models
from raglab.retrieval import Index
from raglab.generation import UsageLedger, generate, GenerationError
from raglab.evaluation import evaluate, evaluate_qa
from raglab.storage import read_json

mimetypes.add_type("font/woff2", ".woff2")
st.set_page_config(page_title="Laya — ответы с источниками", page_icon="🌿", layout="wide")
st.markdown("<style>" + (ROOT / "static/service.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

DATASETS = {"rubq": "Русская Wikipedia", "scifact": "Научные публикации"}
MODE_NAMES = {"bm25": "Поиск по словам", "dense": "Поиск по смыслу", "hybrid": "Смешанный поиск",
              "router": "Laya · выбор поиска", "reranker": "Laya · уточнение результатов", "full": "Laya · полный режим"}
prefs = st.session_state.setdefault("preferences", {"dataset": "rubq", "mode": "hybrid", "profile": "sub2api", "use_api": True, "details": False})
api_keys = st.session_state.setdefault("api_keys", {})
histories = st.session_state.setdefault("histories", {})

def save_preference(name, widget_key):
    prefs[name] = st.session_state[widget_key]

def save_api_key(profile_name, widget_key):
    api_keys[profile_name] = st.session_state[widget_key]

def close_settings():
    st.session_state["settings_open"] = False

def reset_chat():
    st.session_state["histories"] = {**st.session_state["histories"], st.session_state["preferences"]["dataset"]: []}

def benchmark_is_running():
    import psutil
    try:
        state = read_json(ARTIFACTS / "benchmark-running.json")
        process = psutil.Process(state["pid"])
        return abs(process.create_time() - state["create_time"]) < 0.01 and process.is_running()
    except (OSError, ValueError, KeyError, psutil.Error):
        return False

@st.cache_resource
def model_resources():
    return Models()

@st.cache_resource
def index_resource(name, manifest_mtime):
    return Index(prepare(name), model_resources())

@st.fragment(run_every="10s")
def background_progress():
    if not benchmark_is_running():
        st.info("Фоновая обработка завершилась. Обновите страницу, чтобы загрузить результаты.")
        return
    for name in ("rubq", "scifact"):
        directory = ARTIFACTS / "indices" / name
        if (directory / "pending.json").exists() and (directory / "embedding-order.json").exists() and not (directory / "manifest.json").exists():
            done = read_json(directory / "pending.json")["completed"]
            total = len(read_json(directory / "embedding-order.json"))
            st.progress(min(done / total, 1) if total else 0, text=f"Готовим базу: {done:,} из {total:,} фрагментов")
            return
    st.info("Выполняется сравнение качества поиска. Результаты появятся в настройках, в разделе «Отчёты».")

def show_sources(sources):
    for number, hit in enumerate(sources, 1):
        chunk = hit["chunk"] if isinstance(hit, dict) else asdict(hit.chunk)
        text = chunk["text"]
        title = chunk["title"]
        if not title or title.startswith("Wikipedia"):
            title = " ".join(text.split())
        title = title if len(title) <= 76 else title[:73].rsplit(" ", 1)[0] + "…"
        with st.expander(f"[{number}] {title}"):
            st.write(text)
            domain = urlparse(chunk.get("url", "")).hostname or "База знаний"
            st.caption(domain.replace("www.", ""))
            if chunk.get("url"):
                st.link_button("Открыть статью", chunk["url"], icon=":material/open_in_new:")
            if prefs["details"]:
                st.caption(f"Документ {chunk['document_id']} · чанк {chunk['id']}")

@st.dialog("Настройки", width="large", on_dismiss=close_settings)
def settings_dialog():
    search_tab, api_tab, experiments_tab, reports_tab = st.tabs(["Поиск", "Подключение", "Эксперименты", "Отчёты"])
    busy = benchmark_is_running()
    with search_tab:
        st.selectbox("База знаний", list(DATASETS), index=list(DATASETS).index(prefs["dataset"]), format_func=DATASETS.get,
                     key="_dataset", on_change=save_preference, args=("dataset", "_dataset"))
        st.selectbox("Способ поиска", list(MODE_NAMES), index=list(MODE_NAMES).index(prefs["mode"]), format_func=MODE_NAMES.get,
                     key="_mode", on_change=save_preference, args=("mode", "_mode"))
        st.caption("По смыслу — Dense E5. Смешанный поиск — BM25 + Dense + RRF. Laya добавляет локальный выбор или переупорядочивание источников.")
        st.checkbox("Готовить ответ через API", value=prefs["use_api"], key="_use_api", on_change=save_preference, args=("use_api", "_use_api"))
        st.checkbox("Показывать диагностику ответов", value=prefs["details"], key="_details", on_change=save_preference, args=("details", "_details"))
        st.caption("Вопросы обрабатываются независимо: история отображается в чате, но не добавляется к следующему запросу.")
        with st.expander("Управление базой знаний"):
            name = prefs["dataset"]
            manifest_path = DATA / name / "manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                st.caption(f"{manifest['documents']:,} документов · данные сохранены на этом устройстве")
            c1, c2 = st.columns(2)
            build = c1.button("Подготовить базу", disabled=busy, width="stretch")
            rebuild = c2.button("Перестроить индекс", disabled=busy, width="stretch")
            if busy:
                st.caption("Сейчас выполняется фоновый benchmark. Второй тяжёлый процесс временно недоступен.")
            if build or rebuild:
                bar = st.progress(0, text="Подготовка базы")
                try:
                    prepare(name, lambda p, m: bar.progress(p, text=m))
                    resource = index_resource(name, manifest_path.stat().st_mtime_ns)
                    (resource.rebuild if rebuild else resource.build)(lambda p, m: bar.progress(p, text=m))
                    st.session_state["notice"] = "База готова к вопросам"
                    st.rerun()
                except Exception as error:
                    st.error(f"Не удалось подготовить базу ({type(error).__name__}). Проверьте сеть и свободную память.")
    with api_tab:
        available = profiles()
        st.selectbox("Сервис ответов", list(available), index=list(available).index(prefs["profile"]),
                     format_func=lambda name: "sub2api · Luna 5.6" if name == "sub2api" else "RouterAI · DeepSeek Flash",
                     key="_profile", on_change=save_preference, args=("profile", "_profile"))
        profile = available[prefs["profile"]]
        key_name = "_key_" + profile.name
        st.text_input("Ключ API", type="password", value=api_keys.get(profile.name, ""), key=key_name,
                      on_change=save_api_key, args=(profile.name, key_name), help=f"Ключ хранится только в памяти сессии. Можно использовать переменную {profile.key_env}.")
        st.caption(f"Модель: {profile.model}\n\nАдрес: {profile.base_url}")
        usage = UsageLedger().status()
        st.info(f"Запросы без внутреннего лимита. Выполнено обращений: {usage['used']}.")
        st.caption("Запрос отправляется только после вашего действия. Автоматические повторы отключены; расход токенов сохраняется в результатах ответа.")
    with experiments_tab:
        st.write("Сравнение на фиксированных запросах")
        suite = st.radio("Прогон", ["Baseline · полный test", "Laya · 50 вопросов", "QA · 4 вопроса"], key="_suite")
        st.caption("Baseline и Laya считаются локально. QA обращается к выбранному API; завершённые QA-ответы переиспользуются из checkpoint.")
        if st.button("Запустить эксперимент", type="primary", disabled=busy):
            name = prefs["dataset"]
            data_manifest = DATA / name / "manifest.json"
            if not data_manifest.exists() or not (ARTIFACTS / "indices" / name / "manifest.json").exists():
                st.warning("Сначала подготовьте базу в разделе «Поиск».")
            elif suite.startswith("QA") and name != "rubq":
                st.warning("QA с эталонными ответами доступен для русской базы RuBQ.")
            else:
                bar = st.progress(0, text="Подготовка эксперимента")
                try:
                    resource = index_resource(name, data_manifest.stat().st_mtime_ns).build()
                    callback = lambda p, m: bar.progress(min(p, 1), text=m)
                    if suite.startswith("QA"):
                        profile = profiles()[prefs["profile"]]
                        path, _ = evaluate_qa(resource, profile, api_keys.get(profile.name) or None, callback)
                    else:
                        modes = ("bm25", "dense", "hybrid") if suite.startswith("Baseline") else ("hybrid", "router", "reranker", "full")
                        path, _ = evaluate(resource, modes, None if suite.startswith("Baseline") else 50, callback)
                    st.success(f"Готово: {path.name}. Результат доступен во вкладке «Отчёты».")
                except GenerationError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Эксперимент остановлен ({type(error).__name__}). Сохранённые результаты доступны для продолжения.")
    with reports_tab:
        full_report = ROOT / "WORK_REPORT.md"
        if full_report.exists():
            st.download_button("Полный отчёт о работе", full_report.read_bytes(), "Laya-work-report.md", "text/markdown", width="stretch", on_click="ignore")
        reports = sorted((ARTIFACTS / "reports").glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        if reports:
            selected = st.selectbox("Результаты экспериментов", reports, format_func=lambda path: path.stem, key="_report")
            report = read_json(selected)
            st.dataframe(pd.DataFrame(report.get("summary", [])), hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            c1.download_button("Скачать JSON", selected.read_bytes(), selected.name, "application/json", on_click="ignore")
            csv = selected.with_suffix(".csv")
            if csv.exists():
                c2.download_button("Скачать CSV", csv.read_bytes(), csv.name, "text/csv", on_click="ignore")
        else:
            st.caption("Здесь появятся результаты после первого эксперимента.")
    st.divider()
    if st.button("Готово", type="primary", width="stretch"):
        close_settings()
        st.rerun()

with st.container(key="service_header"):
    brand, actions = st.columns([3, 2], vertical_alignment="center")
    brand.markdown('<div class="brand"><img src="/app/static/laya-mark.svg" alt=""><div><div class="brand-name">laya</div><div class="brand-subtitle">ответы с источниками</div></div></div>', unsafe_allow_html=True)
    with actions:
        new_chat, settings = st.columns([1, 1.2])
        new_chat.button("Новый чат", icon=":material/add:", width="stretch", on_click=reset_chat)
        if settings.button("Настройки", icon=":material/tune:", width="stretch"):
            st.session_state["settings_open"] = True

if st.session_state.get("settings_open", False):
    settings_dialog()

if st.session_state.get("notice"):
    st.toast(st.session_state.pop("notice"), icon=":material/check_circle:")

dataset_name = prefs["dataset"]
index = None
index_error = None
manifest_path = DATA / dataset_name / "manifest.json"
if manifest_path.exists():
    try:
        index = index_resource(dataset_name, manifest_path.stat().st_mtime_ns)
        if index.bm25 is None and (index.root / "manifest.json").exists():
            with st.spinner("Открываем базу знаний…"):
                index.build()
    except Exception as error:
        index_error = type(error).__name__
ready = index is not None and index.bm25 is not None
history = histories.setdefault(dataset_name, [])
suggestion = None

if not history:
    label = DATASETS[dataset_name]
    st.markdown(f'<div class="hero"><div class="hero-eyebrow"><span class="status-dot"></span>{escape(label)}</div><h1>Найдите ответ.<br><span class="hero-accent">Проверьте источник.</span></h1><p>Спросите о том, что важно.<br>Мы найдём ответ в документах и покажем, откуда он взят.</p><div class="hero-decoration"><span></span><span></span><span></span><span></span><span></span></div></div>', unsafe_allow_html=True)
    if ready:
        st.markdown('<div class="prompt-intro">С чего начнём?</div>', unsafe_allow_html=True)
        examples = {"rubq": ["В каком году создан Instagram?", "Кто написал «Хижину дяди Тома»?", "Какой океан самый маленький?"],
                    "scifact": ["What affects newborn white matter development?", "What is DNA methylation?", "How do cerebral organoids model brain development?"]}[dataset_name]
        for number, (column, example) in enumerate(zip(st.columns(3), examples)):
            if column.button(example, key=f"suggestion_{number}", width="stretch"):
                suggestion = example
    else:
        st.info("Подготовьте базу знаний в «Настройки → Поиск», чтобы начать диалог.")

if benchmark_is_running():
    background_progress()
if index_error:
    st.warning(f"Базу не удалось открыть ({index_error}). Восстановление доступно в настройках.")

for record in history:
    with st.chat_message("user", avatar=":material/person:"):
        st.write(record["question"])
    with st.chat_message("assistant", avatar=":material/eco:"):
        st.markdown(record["text"])
        for warning in record.get("citation_warnings", []):
            st.warning(warning)
        if record["sources"]:
            st.markdown('<div class="source-count">ИСТОЧНИКИ ОТВЕТА</div>', unsafe_allow_html=True)
            show_sources(record["sources"])
        if prefs["details"]:
            with st.expander("Диагностика ответа"):
                st.json(record["details"])

st.markdown('<div class="service-footer">Ответы по документам · Источники всегда рядом</div>', unsafe_allow_html=True)
question = st.chat_input("Задайте вопрос по документам…", key="chat_prompt", max_chars=4000, disabled=not ready) or suggestion
if question and ready:
    try:
        with st.spinner("Ищем ответ в источниках…"):
            result = index.retrieve(question, prefs["mode"])
            sources = result.hits[:index.settings.context_k]
            profile = profiles()[prefs["profile"]]
            answer = generate(question, sources, profile, api_keys.get(profile.name) or None) if prefs["use_api"] else None
            citation_warnings = []
            if answer and answer.invalid_citations:
                citation_warnings.append("Часть ссылок не соответствует найденным источникам. Проверьте ответ по документам ниже.")
            if answer and answer.missing_citations:
                citation_warnings.append("В ответе не указаны ссылки. Проверьте найденные источники.")
            history.append({"question": question, "text": answer.text if answer else ("Вот что удалось найти по вашему вопросу." if sources else "По этому вопросу источников не найдено."),
                "sources": [asdict(hit) for hit in sources], "citation_warnings": citation_warnings,
                "details": {"mode": LABELS[prefs["mode"]], "route": result.route, "timings_ms": result.timings,
                            "router_confidence": result.confidence, "warnings": result.warnings,
                            "generation_ms": answer.latency_ms if answer else None, "usage": answer.usage if answer else {}}})
        st.rerun()
    except GenerationError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Не удалось выполнить поиск ({type(error).__name__}). Повторите вопрос или проверьте базу в настройках.")

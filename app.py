from dataclasses import asdict
import mimetypes
from urllib.parse import urlparse
import streamlit as st

from raglab.config import ROOT, DATA, ARTIFACTS, LABELS, profiles
from raglab.data import prepare
from raglab.models import Models
from raglab.retrieval import Index
from raglab.traffic import TrafficKnowledge
from raglab.generation import UsageLedger, generate, GenerationError
from raglab.storage import read_json

mimetypes.add_type("font/woff2", ".woff2")
st.set_page_config(page_title="ПДД — личный RAG", page_icon="🚦", layout="wide")
st.markdown("<style>" + (ROOT / "static/service.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

MODE_NAMES = {"bm25": "Поиск по словам", "dense": "Поиск по смыслу", "hybrid": "Смешанный поиск",
              "router": "Laya · выбор поиска", "reranker": "Laya · уточнение результатов", "full": "Laya · полный режим"}
prefs = st.session_state.setdefault("preferences", {"dataset": "pdd", "mode": "hybrid", "profile": "sub2api", "use_api": True, "details": False})
prefs["dataset"] = "pdd"
api_keys = st.session_state.setdefault("api_keys", {})
histories = st.session_state.setdefault("histories", {})

def save_preference(name, widget_key):
    prefs[name] = st.session_state[widget_key]

def save_api_key(profile_name, widget_key):
    api_keys[profile_name] = st.session_state[widget_key]

def close_settings():
    st.session_state["settings_open"] = False

def reset_chat():
    st.session_state["histories"] = {**st.session_state["histories"], "pdd": []}

def benchmark_is_running():
    import psutil
    try:
        state = read_json(ARTIFACTS / "benchmark-running.json")
        process = psutil.Process(state["pid"])
        return abs(process.create_time() - state["create_time"]) < 0.01 and process.is_running()
    except (OSError, ValueError, KeyError, psutil.Error):
        return False

@st.cache_resource
def index_resource(manifest_mtime):
    return Index(prepare("pdd"), Models())

@st.cache_resource
def knowledge_resource(manifest_mtime):
    return TrafficKnowledge(prepare("pdd").documents, DATA / "pdd")

def show_images(assets, knowledge):
    if not assets or knowledge is None:
        return
    for start in range(0, len(assets), 3):
        for column, asset in zip(st.columns(min(3, len(assets) - start)), assets[start:start + 3]):
            with column:
                try:
                    photo = asset["kind"] == "real_world_photo_crop"
                    label = f"{asset['code']} · {asset['name']}"
                    st.image(knowledge.image_bytes(asset), width=180,
                             caption=("Фото с дороги · " if photo else "") + label, alt=label)
                    if photo:
                        st.caption("Историческое фото RTSD; значения и правила — по нормативным источникам.")
                    st.markdown(f"[{'RTSD' if photo else asset.get('standard', 'Каталог знаков')}]({asset['source_url']}) · источник [{asset['source_number']}]")
                except (OSError, ValueError):
                    st.warning(f"Изображение знака {asset['code']} недоступно. Текст источника сохранён ниже.")

def show_sources(sources):
    for number, hit in enumerate(sources, 1):
        chunk = hit["chunk"] if isinstance(hit, dict) else asdict(hit.chunk)
        title = chunk["title"] or " ".join(chunk["text"].split())
        title = title if len(title) <= 85 else title[:82].rsplit(" ", 1)[0] + "…"
        with st.expander(f"[{number}] {title}"):
            st.write(chunk["text"])
            domain = urlparse(chunk.get("url", "")).hostname or "База ПДД"
            st.caption(domain.replace("www.", ""))
            if chunk.get("url"):
                st.link_button("Открыть источник", chunk["url"], icon=":material/open_in_new:")
            if prefs["details"]:
                st.caption(f"Документ {chunk['document_id']} · фрагмент {chunk['id']}")

@st.dialog("Настройки", width="large", on_dismiss=close_settings)
def settings_dialog():
    search_tab, api_tab, about_tab = st.tabs(["Поиск", "Подключение", "О базе"])
    busy = benchmark_is_running()
    with search_tab:
        st.selectbox("Способ поиска", list(MODE_NAMES), index=list(MODE_NAMES).index(prefs["mode"]), format_func=MODE_NAMES.get,
                     key="_mode", on_change=save_preference, args=("mode", "_mode"))
        st.caption("Смешанный поиск объединяет слова и смысл. Laya доступна как дополнительный режим; её польза на ПДД пока не измерена.")
        st.checkbox("Готовить ответ через API", value=prefs["use_api"], key="_use_api", on_change=save_preference, args=("use_api", "_use_api"))
        st.checkbox("Показывать диагностику ответов", value=prefs["details"], key="_details", on_change=save_preference, args=("details", "_details"))
        st.caption("Каждый вопрос ищется отдельно. История предыдущих вопросов не передаётся генератору.")
        with st.expander("Управление базой ПДД"):
            manifest_path = DATA / "pdd" / "manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                st.caption(f"{manifest['documents']:,} документов · снимок {manifest.get('snapshot_date', 'сохранён локально')}")
            c1, c2 = st.columns(2)
            build = c1.button("Подготовить базу", disabled=busy, width="stretch")
            rebuild = c2.button("Перестроить индекс", disabled=busy, width="stretch")
            if busy:
                st.caption("Выполняется фоновая обработка базы.")
            if build or rebuild:
                bar = st.progress(0, text="Подготовка ПДД")
                try:
                    prepare("pdd", lambda p, m: bar.progress(p, text=m))
                    resource = index_resource(manifest_path.stat().st_mtime_ns)
                    (resource.rebuild if rebuild else resource.build)(lambda p, m: bar.progress(p, text=m))
                    st.session_state["notice"] = "База ПДД готова"
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
                      on_change=save_api_key, args=(profile.name, key_name), help=f"Ключ хранится в памяти сессии. Можно задать переменную {profile.key_env}.")
        st.caption(f"Модель: {profile.model}\n\nАдрес: {profile.base_url}")
        st.info(f"Запросы без внутреннего лимита. Выполнено обращений: {UsageLedger().status()['used']}.")
        st.caption("Запрос отправляется только после вашего действия. Автоматические повторы отключены.")
    with about_tab:
        manifest_path = DATA / "pdd" / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            coverage = manifest.get("coverage", {})
            st.write("ПДД России, дорожные знаки, правила их применения и связанные нормы об ответственности.")
            st.caption(f"Снимок: {manifest.get('snapshot_date', '—')} · редакция ПДД: {manifest.get('revision', {}).get('pdd', '—')}")
            st.write(f"{manifest['documents']:,} документов · {coverage.get('sign_images', 0)} эталонных знаков · {coverage.get('real_photos', 0)} фото")
        st.caption("Картинки берутся из локального каталога. Поиск по номеру и названию знака работает с текстовым вопросом; распознавание загруженных фото пока не подключено.")
        for label, name in [("О составе и источниках базы", "docs/PDD_DATASET.md"),
                            ("Отчёт о версии ПДД", "docs/PDD_SERVICE.md"),
                            ("Историческое сравнение Laya", "BENCHMARKS.md")]:
            path = ROOT / name
            if path.exists():
                st.download_button(label, path.read_bytes(), path.name, "text/markdown", width="stretch", on_click="ignore")
    st.divider()
    if st.button("Готово", type="primary", width="stretch"):
        close_settings()
        st.rerun()

with st.container(key="service_header"):
    brand, actions = st.columns([3, 2], vertical_alignment="center")
    brand.markdown('<div class="brand"><img src="/app/static/laya-mark.svg" alt=""><div><div class="brand-name">пдд</div><div class="brand-subtitle">личный RAG по ПДД России</div></div></div>', unsafe_allow_html=True)
    with actions:
        new_chat, settings = st.columns([1, 1.2])
        new_chat.button("Новый чат", icon=":material/add:", width="stretch", on_click=reset_chat)
        if settings.button("Настройки", icon=":material/tune:", width="stretch"):
            st.session_state["settings_open"] = True

if st.session_state.get("settings_open", False):
    settings_dialog()
if st.session_state.get("notice"):
    st.toast(st.session_state.pop("notice"), icon=":material/check_circle:")

index = knowledge = None
index_error = None
manifest_path = DATA / "pdd" / "manifest.json"
if manifest_path.exists():
    try:
        modified = manifest_path.stat().st_mtime_ns
        index = index_resource(modified)
        if index.bm25 is None and (index.root / "manifest.json").exists():
            with st.spinner("Открываем базу ПДД…"):
                index.build()
        knowledge = knowledge_resource(modified)
    except Exception as error:
        index_error = type(error).__name__
ready = index is not None and index.bm25 is not None and knowledge is not None
history = histories.setdefault("pdd", [])

if not history:
    st.markdown('<div class="hero hero-pdd"><h1>Разберёмся в ПДД.</h1></div>', unsafe_allow_html=True)
    if not ready:
        st.info("Подготовьте ПДД в «Настройки → Поиск → Управление базой ПДД».")
if index_error:
    st.warning(f"Базу не удалось открыть ({index_error}). Восстановление доступно в настройках.")

for record in history:
    with st.chat_message("user", avatar=":material/person:"):
        st.write(record["question"])
    with st.chat_message("assistant", avatar=":material/traffic:"):
        show_images(record.get("images", []), knowledge)
        st.markdown(record["text"])
        for warning in record.get("citation_warnings", []):
            st.warning(warning)
        if record["sources"]:
            st.markdown('<div class="source-count">ИСТОЧНИКИ ОТВЕТА</div>', unsafe_allow_html=True)
            show_sources(record["sources"])
        if prefs["details"]:
            with st.expander("Диагностика ответа"):
                st.json(record["details"])

question = st.chat_input("Спросите о ПДД или дорожном знаке…", key="chat_prompt", max_chars=4000, disabled=not ready)
if question and ready:
    try:
        with st.spinner("Ищем в ПДД и разбираем правила…"):
            result = index.retrieve(question, prefs["mode"])
            sources = knowledge.sources(question, result.hits, index.settings.context_k)
            profile = profiles()[prefs["profile"]]
            available_images = knowledge.images(question, sources)
            answer = generate(question, sources, profile, api_keys.get(profile.name) or None, purpose="pdd-chat", images=available_images) if prefs["use_api"] else None
            citation_warnings = []
            if answer and answer.invalid_citations:
                citation_warnings.append("Часть ссылок не соответствует источникам. Проверьте ответ по документам ниже.")
            if answer and answer.missing_citations:
                citation_warnings.append("В ответе нет ссылок. Проверьте найденные источники.")
            history.append({"question": question, "text": answer.text if answer else ("Найденные знаки и правила:" if sources else "По этому вопросу источников не найдено."),
                "sources": [asdict(hit) for hit in sources], "images": knowledge.images(question, sources, answer.text if answer else None),
                "citation_warnings": citation_warnings,
                "details": {"mode": LABELS[prefs["mode"]], "route": result.route, "timings_ms": result.timings,
                            "router_confidence": result.confidence, "warnings": result.warnings,
                            "generation_ms": answer.latency_ms if answer else None, "usage": answer.usage if answer else {}}})
        st.rerun()
    except GenerationError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Не удалось выполнить поиск ({type(error).__name__}). Повторите вопрос или проверьте базу в настройках.")

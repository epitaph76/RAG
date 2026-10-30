# Сборка подробного отчёта

Готовый PDF: [PDD_RAG_Detailed_Report_RU.pdf](../../output/pdf/PDD_RAG_Detailed_Report_RU.pdf). [Текстовая версия](../PDD_RAG_DETAILED_REPORT_RU.md).

Отчёт описывает состояние приложения `a0d2926` и снимок корпуса от 30.10.2026. Дата отчёта - 31.10.2026. Исторические метрики не пересчитывались для документа; источники каждого результата указаны в тексте.

Воспроизводимая сборка не требует корпуса, моделей или API-ключей. Нужны исходные главы, включённые в репозиторий шрифты, два сохранённых скриншота и небольшой JSON с проверенными фактами.

Из корня проекта в отдельном окружении:

```powershell
py -3.12 -m venv .report-venv
.\.report-venv\Scripts\python.exe -m pip install -r docs/report_assets/requirements.txt
.\.report-venv\Scripts\python.exe -X utf8 scripts/build_project_report.py
```

Исходники глав: `docs/report_source/*.md`. Генератор создаёт PDF в `output/pdf`, объединённый Markdown в `docs` и технические схемы SVG в `docs/report_assets`. Схемы и графики остаются векторными в PDF, текст можно выделять и искать. Иллюстрации интерфейса взяты из уже сохранённых скриншотов, новые изображения знаков не генерировались.

Для проверки PDF можно использовать Poppler (`pdftoppm`, `pdfinfo`), pypdf и pdfplumber. Они не нужны для обычной сборки и не входят в зависимости приложения. Временные изображения проверки находятся в игнорируемом `tmp/pdfs`.

## Шрифты и происхождение

- Noto Sans Regular/Bold: [официальный репозиторий Noto](https://github.com/notofonts/noto-fonts/tree/main/hinted/ttf/NotoSans), лицензия в `fonts/NotoSans-LICENSE.txt`.
- JetBrains Mono Medium: [официальный репозиторий JetBrains](https://github.com/JetBrains/JetBrainsMono/tree/master/fonts/ttf), лицензия в `fonts/JetBrainsMono-OFL.txt`.
- Скриншоты: `docs/pdd-home.jpg` и `docs/pdd-sign.jpg`, работающая локальная версия приложения.
- Диаграммы: по коду проекта. Графики Laya: по `BENCHMARKS.md` и `benchmarks/results.json`. Таблица RRF явно обозначена как учебный пример.

Сборка не обращается к API генерации, не меняет индексы и не читает секреты.

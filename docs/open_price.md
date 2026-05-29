# `open_price.py` — описание модуля

Документ отражает **текущую** реализацию [`open_price.py`](../open_price.py): прогноз цены открытия SPX на RTH-сессию (NY), вспомогательная диагностика гэпа и **режим волатильности для Summary** (`forecast_vix_session_level`). Источник истины при расхождении — сам код.

---

## 1. Назначение

1. **`open_pred`** — оценка открытия SPX на первом RTH-баре даты `plan_for`, если минутные данные ещё не содержат этот бар; иначе подстановка **фактического Open** первого RTH-бара из CSV (см. §10).
2. **`Components`** для диагностики: baseline по истории SPX, премаркетная оценка ES к 09:30, свежий ES↔SPX базис, смесь baseline/`spx_es_level`, ограничение гэпа по робастной сигме.
3. **Режим волатильности и «VIX total» для Summary** — через `forecast_vix_session_level` (не участвует в числовой формуле `open_pred`, только в полях результата / совместимости с TradePlan).

Модуль **не** использует обучаемые веса и **не** зависит от `backtest_open_weights`.

---

## 2. Точка входа и поток данных

### 2.1 `predict_spx_open(plan_for, пути к CSV, ...)`

1. Читает **SPX 1m** (`_read_1min_csv`), затем проверяет **`_actual_spx_rth_open_from_first_bar`**: если на `plan_for` уже есть RTH-бары (09:30–15:59 NY), **итоговый `open_pred`** позже заменяется на Open первого такого бара (модель всё равно считается для компонент и метаданных).
2. По возможности загружает: ES 5m (обязателен для основной ветки ES), ES 15m/1m (в **`compute_components` не используются** — остаются в сигнатуре для совместимости API и вспомогательных функций регрессии в файле), ES 1m RTH (базис), VIX 5m/15m/1m, VIX9D/VIX3M 5m, дневной VIX.
3. Вызывает **`compute_components`** → получает `Components`, `ref_date`, `ref_close`.
4. Считает **динамическую смесь** `baseline_open` и `spx_es_level` (§6), затем **clamp** гэпа относительно `baseline_gap` и робастной `open_sigma` (§7).
5. При наличии фактического первого бара — подменяет `open_pred`, пересчитывает gap Z-score и помечает `open_pred_source` в meta.
6. Возвращает **`OpenPriceResult`**: `open_pred`, `ref_close`, `regime` (из session-level VIX), веса смеси, словарь компонент, `vix_metrics`, расширенный `meta`.

### 2.2 `compute_components`

Использует только: **`spx_1m`**, **`es_5m`**, **`es_1m_rth`**, **`vix_5m`**, **`vix_15m`**, **`vix_1m`**, **`vix9d_5m`**, **`vix3m_5m`** (последний в вызове передаётся, но **`forecast_vix_session_level` внутри использует VIX 5m и VIX9D 5m**), **`vix_daily`** (не используется в текущей сборке компонент — см. код).

Последовательность:

- **`compute_baseline(spx_1m, plan_for)`** → ref_close, baseline_gap, baseline_open, meta с сигмой.
- **`compute_es_pred_930_ema(es_5m, plan_for, vix_1m=...)`** — премаркет ES + опциональная поправка по **VIX 1m** (последние 3 бара до 09:30).
- **`compute_fresh_basis(es_1m_rth, spx_1m, plan_for)`** — только последняя завершённая сессия.
- **`spx_es_level = es_pred_930 − basis_level_k × fresh_basis`**, где **`basis_level_k = _basis_level_multiplier(fresh_basis)`** и в коде для конечного базиса всегда **1.0** (`FRESH_BASIS_LEVEL_K_BASE`).
- **`forecast_vix_session_level(..., premkt_weight=0.5)`** → `vix_session_total`, строковый `regime` для Summary (поля `Components.vix_regime`, `vix_session_forecast`).

Функции **`_build_es_regression_features`**, **`_map_es_to_spx_gap`**, **`_compute_es_smooth_*`** и др. в файле **не входят** в основной путь `predict_spx_open` / `compute_components`; они остаются утилитарными/историческими блоками.

---

## 3. Входные файлы (контракт CSV)

| Роль | Файл (типичное имя) |
|------|---------------------|
| Обязательный: SPX минутки | `SPX_1min_90d_ibkr.csv` — первая колонка время; нужны open/high/low/close |
| Обязательный для модели ES | `ES_5min_90d.csv` |
| Базис ES↔SPX | `ES_1min_90d_RTH.csv` (и SPX 1m для синхронных баров) |
| Поправка ES по воле | `VIX_1min_90d.csv` — последние 3 минутных бара премаркета для множителя к `close_last` |
| Summary: режим / VIX total | `VIX_5min_90d.csv`, опционально `VIX9D_5min_90d.csv` |

Опционально подгружаются ES 15m / ES 1m полный день — **на результат `open_pred` в текущей связке `compute_components` не влияют**.

Данные предполагаются в «NY-логике» как в загрузчике проекта: колонка `Datetime` часто **naive** после нормализации таймзоны.

---

## 4. Загрузка CSV (`_read_1min_csv`, `_read_5min_csv`, `_read_15min_csv`)

- Проверка существования файла; пустой DF допускается только до последующих проверок.
- Первая колонка или явная `Datetime` → приводится к `Datetime`.
- Имена OHLC приводятся к `Open`, `High`, `Low`, `Close`; при наличии — `Volume`.
- `Datetime` парсится, строки без времени отбрасываются; сортировка по времени.
- OHLC приводятся к числу; строки без валидного `Close` удаляются.

---

## 5. Baseline: `compute_baseline`

**Цель:** робастный «типичный гэп» по истории SPX до `plan_for`.

1. **`ref_date`** — последняя календарная дата сессии в файле **строго до** `plan_for`.
2. **`ref_close`** — Close последнего бара на `ref_date` с временем **строго до 16:00** (`_get_last_bar_before_time(..., strict=True)`).
3. Для каждой пары последовательных торговых дней из списка дат файла: Open в **09:30** сегодня минус Close «конца вчерашней сессии» (последний бар вчера до 16:00) → список гэпов; минимум **8** пар, иначе ошибка.
4. **`global_gap_median`** = медиана всех гэпов.
5. **День недели:** если для того же weekday как у `plan_for` набралось **≥ `MIN_DOW_COUNT` (5)** гэпов:  
   `baseline_gap = BASELINE_SHRINKAGE_GLOBAL * global + BASELINE_SHRINKAGE_DOW * dow_median`  
   (**0.3 / 0.7**). Иначе `baseline_gap = global_gap_median`.
6. **`baseline_open = ref_close + baseline_gap`**.
7. **Робастная сигма гэпа:** MAD относительно `global_gap_median`, **`open_sigma = 1.4826 * MAD`**, при вырождении — fallback на std гэпов; дополнительно в meta: IQR, счётчики истории.

---

## 6. ES к 09:30: `compute_es_pred_930_ema`

**Окно премаркета на `plan_for`:** бары ES 5m с **04:00 ≤ t < 09:30** (`_get_premarket_bars_5m`). Нужно минимум 2 бара.

**EMA:** три канала длиной 5, 9, 10; инициализация первым Close; стандартная рекурсия \(α = 2/(N+1)\).

**Базовый уровень:**

- `es_base = 0.5·ema5 + 0.3·ema9 + 0.2·ema10`.
- Стек EMA: «bull» если ema5 > ema9 > ema10; «bear» если все три строго убывают; иначе «mixed».
- `trend_adj = 0.25 * (ema5 − ema10)` только в состояниях bull/bear; иначе 0.
- **`es_pred_930_no_vix = es_base + trend_adj`**.

**Микропоправка по VIX 1m:** если на `plan_for` есть ≥3 минутных Close VIX строго до 09:30:

- Среднее по ним → режим по порогам **`VIX_REGIME_LOW/NORMAL_HI/HIGH_HI`** (15 / 19 / 23).
- Множитель: normal/low → 0; high → **0.2**; extreme → **0.5**.
- **`es_pred_930 = es_pred_930_no_vix + vix_multiplier * (close_last − es_pred_930_no_vix)`**, где `close_last` — Close последнего премаркет-бара ES 5m.

---

## 7. Свежий базис: `compute_fresh_basis`

Только **последняя завершённая** дата сессии в SPX до `plan_for`.

- Окно **09:30 ≤ t < 09:35:** для каждого общего по времени бара `basis_t = ES_close − SPX_close`; среднее → **`fresh_basis_open`**.
- Окно **09:30 ≤ t < 10:00:** то же → **`fresh_basis_30m`**.
- **`fresh_basis = 0.7 * fresh_basis_open + 0.3 * fresh_basis_30m`**.

При отсутствии пар баров или данных возвращается NaN с кодом ошибки в meta.

---

## 8. Уровень SPX, имплицитный из ES: `spx_es_level`

При конечных `es_pred_930` и `fresh_basis`:

```text
spx_es_level = es_pred_930 − fresh_basis
```

(коэффициент при базисе в коде зафиксирован как 1.0 через `_basis_level_multiplier`.)

---

## 9. Динамическая смесь baseline и ES (`_open_blend_weights`)

Ввод: **`baseline_open`**, **`spx_es_level`**.  
`diff = baseline_open − spx_es_level`.

| Условие | Вес baseline | Вес spx_es_level | Примечание |
|---------|----------------|------------------|------------|
| не оба конечны | 0.10 | 0.90 | режим «default» в коде для неконечных входов фактически не используется для смеси — см. ниже |
| `diff > BASELINE_ES_DOMINANT_DIFF` (**20.0**) | **0.00** | **1.00** | в meta режим назван `es_95_baseline_above_30` (историческое имя), по факту — полное доминирование ветки ES |
| `0 < diff ≤ 20` | 0.15 | 0.85 | `es_dominant_baseline_above` |
| `diff ≤ 0` | **0.10** | **0.90** | `default` |

Если **`spx_es_level`** не число — смесь не применяется: **`open_pred_raw = baseline_open`**, веса baseline=1, spx_es_level=0.

---

## 10. Clamp гэпа и метрики неопределённости

1. **`pred_gap = open_pred_raw − ref_close`** (до подмены фактическим баром).
2. **`max_dev = CLAMP_K_SIGMA * open_sigma`** (**CLAMP_K_SIGMA = 3**).
3. **`pred_gap_clamped`** ограничивается интервалом **`[baseline_gap − max_dev, baseline_gap + max_dev]`**.
4. **`open_pred = ref_close + pred_gap_clamped`** (модельная величина сохраняется как `open_pred_model`, если позже подставлен фактический Open).

Дополнительно:

- **`gap_z = (open_pred − ref_close) / open_sigma`** (после финального open_pred),
- **`GAP_Z_THRESHOLD = 0.6`** → `gap_is_significant`, `gap_sign` ∈ {−1, 0, +1}.

---

## 11. Подстановка фактического Open (`_actual_spx_rth_open_from_first_bar`)

Если в SPX 1m на дату `plan_for` есть минутные бары с временем **09:30–15:59**, берётся **Open самого раннего** такого бара → финальный **`open_pred`**, **`open_pred_source = "spx_1m_first_rth_bar"`**, в meta добавляются время бара и сохранённое модельное значение.

---

## 12. Режим волатильности для Summary: `forecast_vix_session_level`

**Не влияет** на формулу смеси/clamp открытия.

Алгоритм (упрощённо):

1. По последним **`n_sessions=10`** завершённым дням до `plan_for`: для **VIX 5m** и при наличии **VIX9D 5m** считается медиана Close в RTH **[09:30, 16:00)** на день; из медиан по дням берётся медиана серии; если оба ряда есть — **`mean_vix`** = среднее двух таких медиан медиан, иначе доступный ряд.
2. Премаркет **`plan_for`:** среднее Close по последним **12** пятиминутным барам VIX до 09:30 → **`premkt_val`** (или fallback через аргумент `vix_premkt`).
3. **`vix_session_total = (1 - premkt_weight) * mean_vix + premkt_weight * premkt_val`**, где:
   - в `compute_components` используется **`premkt_weight=0.5`**;
   - дефолт `forecast_vix_session_level` (если вызван напрямую) — **`premkt_weight=0.3`**.
4. Режим по тем же порогам **`VIX_REGIME_*`**, что и для ES-блока VIX.

Классификатор **`classify_vix_regime`** и **`_forecast_vix_session_volatility`** в модуле остаются для других сценариев; **основной экспорт в результат для TradePlan — из `forecast_vix_session_level` через `compute_components`.**

---

## 13. CLI (`main`)

Парсинг `--plan-for`, сбор путей из `--data-dir`, вызов `predict_spx_open`, печать компонент и сохранение **`open_prediction_<plan_for>.json`** в каталог скрипта (`open_pred`, `ref_close`, источник open).

---

## 14. Константы (справочно)

| Имя | Значение | Назначение |
|-----|----------|------------|
| `W_BASELINE` / `W_SPX_ES_LEVEL` | 0.1 / 0.9 | Базовая смесь при `diff ≤ 0` |
| `W_BASELINE_ES_DOMINANT` / `W_SPX_ES_LEVEL_DOMINANT` | 0.15 / 0.85 | При baseline выше ES-уровня |
| `W_BASELINE_ES_ONLY` / `W_SPX_ES_LEVEL_ES_ONLY` | 0 / 1 | Большой разнос baseline vs ES |
| `BASELINE_ES_DOMINANT_DIFF` | 20 | Порог «большого разноса» |
| `BASELINE_SHRINKAGE_GLOBAL` / `DOW` | 0.3 / 0.7 | День недели в baseline_gap |
| `MIN_DOW_COUNT` | 5 | Минимум наблюдений для dow-компоненты |
| `CLAMP_K_SIGMA` | 3 | Допуск гэпа в сигмах относительно baseline_gap |
| `GAP_Z_THRESHOLD` | 0.6 | Порог значимости Z-score гэпа |
| `VIX_REGIME_LOW` … `HIGH_HI` | 15, 19, 23 | Единые пороги режимов VIX |

---

*При изменении весов смеси, порога 20 пунктов, логики базиса или session-VIX обновляйте этот файл.*

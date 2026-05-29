# `update_TradePlan.py` — внутридневное обновление Trade Plan

Документ описывает **текущую** реализацию [`update_TradePlan.py`](../update_TradePlan.py): загрузка свежих RTH-баров, пересчёт bias/chop, пересборка строк Trade Plan и запись версионированного файла **`update_TradePlan_<дата>_<i>.xlsx`**. При расхождениях источник истины — код.

Связанные документы: [`chop_algorithm.md`](chop_algorithm.md) (§22 intraday chop), [`build_TradePlan.md`](build_TradePlan.md), [`manage_modules.md`](manage_modules.md) (команда `update`).

---

## 1. Назначение

Модуль дополняет утренний **`SPX_TradePlan_<дата>.xlsx`** в ходе RTH-сессии:

- подтягивает актуальные 1m-бары SPX (и ES для bias-признаков);
- фиксирует **фактические** bias/chop по завершённым RTH-сегментам;
- прогнозирует **оставшиеся** стадии bias и chop (Ridge, как в `bias.py` / `chop.py`);
- пересчитывает **Expected Range**, targets, Chop и адаптирует **Entry2–4** для активного сценария;
- сохраняет **`update_TradePlan_<дата>_<i>.xlsx`** (копия предыдущей книги update, либо утренней книги + обновлённые Summary / Trade Plan).

Утренний **`SPX_TradePlan_*`** модуль **не перезаписывает**.

---

## 2. Запуск

```text
python update_TradePlan.py --plan-date YYYY-MM-DD --data-dir data_spx
```

| Аргумент | Описание |
|----------|----------|
| `--plan-date` | Дата сессии `plan_for` (обязательный) |
| `--data-dir` | Каталог данных (по умолчанию `data_spx` от корня проекта) |
| `--bias-table` | Путь к `bias_regression_table.csv.gz` (по умолчанию корень проекта) |
| `--skip-fetch` | Не вызывать IBKR; ожидаются готовые `*_last_data.csv` |
| `--source-xlsx` | Исходный утренний xlsx (по умолчанию `SPX_TradePlan_<дата>.xlsx`) |
| `--ib-port` | Порт TWS (по умолчанию 7496) |

Оркестратор:

```text
python manage.py update YYYY-MM-DD
```

---

## 3. Входные и выходные файлы

### 3.1 Данные (`data_dir`)

| Файл | Роль |
|------|------|
| `SPX_last_data.csv` | Свежие 1m-бары текущей сессии (обязателен) |
| `ES_last_data.csv` | ES для merge в bias-признаки (опционально) |
| `SPX_1min_90d_ibkr.csv` | Исторический SPX; бары `plan_for` заменяются на last_data |
| `ES_1min_90d.csv` | То же для ES |
| `SPX_1d_1y.csv` | Дневной SPX для ref_close / ATR |

Загрузка last_data: **`fresh_data_loader.run_fetch`** (если не `--skip-fetch`). Интервал опроса — `RECOMMENDED_INTRADAY_FETCH_CLOCK_ET` в `fresh_data_loader.py`.

### 3.2 Кэши моделей

- `bias_regression_table.csv.gz` — таблица bias для staged Ridge;
- `chop_regression_table.csv.gz` — через `chop.load_chop_regression_table_cache()` (медианы `chop_dur_20_k`, `session_range_k`, similarity).

### 3.3 Excel и debug

| Путь | Описание |
|------|----------|
| `SPX_TradePlan_<дата>.xlsx` | Источник для копии (утренний план) |
| `update_TradePlan_<дата>_<i>.xlsx` | **Выход** update (каждый запуск — новый файл) |
| `bias_last_prediction_debug.json` | Audit bias/chop/intraday (перезаписывается каждым запуском) |

Каталог планов: `TRADEPLAN_PLANS_DIR` (env) или корень проекта.

---

## 4. Фазы по времени последнего бара (ET)

Фаза определяется **`_detect_phase(last_bar_time)`** по последнему `Datetime` в `SPX_last_data.csv`:

| Условие (время последнего бара) | Фаза | Завершённые RTH-сегменты | Bias Ridge | Chop staged |
|----------------------------------|------|--------------------------|------------|-------------|
| ≤ 12:00 | `stage2` | 1 (09:30–11:00) | факт `*_1` → прогноз 2→3→4 | факт seg1 → прогноз 2→3→4 |
| (12:00, 13:30] | `stage3` | 1–2 | факт `*_1,*_2` → прогноз 3→4 | факт seg1–2 → прогноз 3→4 |
| > 13:30 | `stage4` | 1–3 | факт `*_1..*_3` → прогноз 4 | факт seg1–3 → прогноз 4 |

Границы обрезки RTH-блоков для **факта** (совпадают с bias):

- stage2: inclusive до **11:00**;
- stage3: exclusive до **12:30** (сегмент 2 «почти» закрыт);
- stage4: exclusive до **14:00**.

Сегменты RTH: **90 + 90 + 90 + 120** минут (как `bias.py` / `chop.py`).

---

## 5. Пайплайн `run_update()`

Сжатая последовательность:

1. **`fresh_data_loader.run_fetch`** (если не `--skip-fetch`).
2. Чтение **`SPX_last_data.csv`**, определение **фазы** и **`actual_open`** (Open первого бара).
3. **Merge** `SPX_last_data` / `ES_last_data` в исторические CSV (`_merge_session_bars`).
4. **Bias intraday** ([§6](#6-bias-intraday)):
   - факт сегментов из обрезанного RTH-блока;
   - признаки на merged SPX/ES;
   - цепочка `_predict_stage` для dir/strength;
   - `bias_combined_1..4`, `bias_for_chop` → `audit_payload`.
5. **Chop intraday** ([§7](#7-chop-intraday)):
   - blended actuals `chop_dur_k` по завершённым сегментам;
   - `predict_session_chop` со staged actuals / predict stages;
   - **A2** `compute_effective_intraday_chop_segments` + guardrail;
   - **reapply** EM/targets/zones (`reapply_intraday_chop_session_result`).
6. Запись **`bias_last_prediction_debug.json`**.
7. **Rebuild Trade Plan** ([§8](#8-trade-plan-rebuild-and-adaptive-layers)):
   - `rebuild_trade_plan_intraday` → `assemble_trade_plan`;
   - `apply_intraday_scenario_adaptive_layer` (Entry2–4, active scenario).
8. **Excel**: копия **последнего** `update_TradePlan_<дата>_<i>.xlsx` (если есть) **или** `SPX_TradePlan_*` → новый `update_TradePlan_<дата>_<i+1>.xlsx`, правки Summary + Trade Plan.

---

## 6. Bias intraday

### 6.1 Факт сегментов

`bias_mod._bias_segment_targets_from_block(block_trimmed, atr_d)` → inject `bias_dir_k`, `bias_strength_k` для завершённых k.

### 6.2 Признаки

- Базовые: `_bias_prepare_current_features_for_date`;
- Перезапись merged-потоком: `get_bias_regression_features_for_date(..., spx_1m=combined_spx, es_1m=combined_es)`.

### 6.3 Staged Ridge

`_predict_stage(regression_df, stage, kind, ...)` — обучение на `stage_df`, предсказание с extra-признаками из фактов предыдущих сегментов (логика как в `bias.py`).

### 6.4 Combined для chop и Trade Plan

`bias_prediction_combined_audit` → `bias_combined_1..4`, **`bias_for_chop`** (scalar для chop similarity и `assemble_trade_plan`).

Тексты **Bias Trend** в Excel: `_staged_bias_texts(plan_for, audit_payload=...)` — **из памяти того же запуска**, без повторного чтения JSON.

---

## 7. Chop intraday

### 7.1 Blended actuals

`chop_intraday_blended_actuals_from_trimmed_session_block(blk_ch, reg_chop, n_complete_segments)`:

- flat-coverage 30m/20m на срезе сегмента;
- blend с **медианой** `chop_dur_20_k` из кэша ([chop §4.3 / §22.1](chop_algorithm.md#221-blended-segment-actuals-from-live-bars)).

### 7.2 `predict_session_chop`

Ключевые аргументы:

- `open_pred = actual_open`;
- `session_bias_override = (bias_dir_4, bias_strength_4)` при наличии;
- `bias_for_chop_combined`;
- `intraday_chop_segment_actuals`, `intraday_predict_stages`:
  - если blended actuals **полные** для фазы — якорь + predict `[2,3,4]` / `[3,4]` / `[4]`;
  - иначе — **полная цепочка** `[1,2,3,4]` без якоря на каждом запуске;
- `spx_1m` = RTH merged; `spx_1m_for_regression_cache` = полный merged день.

### 7.3 Layer A2 + guardrail

`compute_effective_intraday_chop_segments(chop_result, chop_actuals)`:

- смешивает факт завершённых сегментов и модель для будущих;
- если средний факт chop **высокий** — поднимает floor на незавершённых сегментах (**+chop day**):
  - mean ≥ **70%** → min **55%** на сегмент;
  - mean ≥ **60%** → min **45%**.

`effective_chop_pct` → взвешенная сессионная доля после guardrail.

### 7.4 Reapply EM / targets

При успешном A2:

1. `_build_intraday_pool_anchors` — `session_range_pts` по завершённым сегментам для similarity ([chop §14.2](chop_algorithm.md#142-intraday-session_range-anchors-and-post-filter)).
2. **`reapply_intraday_chop_session_result`** — пересчёт `mean_range_spx_*`, scenario targets, chop-zones с:
   - `effective_chop_pct` и сегментными процентами после guardrail;
   - `intraday_pool_anchors`;
   - envelope **targets ⊇ Expected Range** (`enforce_chop_targets_cover_expected_range`).

Audit: поля `effective_intraday_chop_pct`, `chop_guardrail`, `intraday_chop_reapply`, `intraday_pool_anchors` в `bias_last_prediction_debug.json`.

---

## 8. Trade Plan rebuild and adaptive layers

### 8.1 `rebuild_trade_plan_intraday`

Полный вызов **`assemble_trade_plan`** (`build_TradePlan.py`) с:

- `open_pred_override = actual_open`;
- `chop_result` после reapply;
- `spx_1m_rth` = merged RTH;
- `bias_score` из `bias_for_chop_combined`;
- Key levels **не пересчитываются** в update: колонка `Key levels` берётся из source xlsx (последний update или утренний план) и сохраняется как есть.

Возвращает две строки Trade Plan + **`plan_meta`** + строки Summary (EM / Expected Range).

### 8.2 Adaptive layer (active scenario)

`apply_intraday_scenario_adaptive_layer(refresh_rows, morning_rows, ...)` — **только активная** строка (bull или bear по `bias_combined_1` / утреннему Scenario):

| Слот | Поведение |
|------|-----------|
| **1** | Entry / Target1 из **утреннего** плана (память) |
| **2–4** | Predictive-цепочка пересчитывается частично: меняется только «следующий» slot entry, остальные слоты продолжаются по прежним правилам |

### 8.2.1 Правило «следующий entry после update»

На каждом intraday update выбирается `anchor_slot = n_done + 1`:

- stage2 (`n_done=1`) → `anchor_slot=2` → обновляется **`Entry point_2`**
- stage3 (`n_done=2`) → `anchor_slot=3` → обновляется **`Entry point_3`**
- stage4 (`n_done=3`) → `anchor_slot=4` → обновляется **`Entry point_4`**

Для активного сценария слоты **строго меньше** `anchor_slot` сохраняются из source xlsx (предыдущего update), а на `anchor_slot` entry якорится на **последний доступный `Close`** текущих данных.

### 8.2.2 Двухвариантный якорный entry (move up / move down)

Начиная с первого intraday update, ячейка `Entry point_{anchor_slot}` выводится в виде:

```text
if move up <last_close> → t1
if move down <last_close> → t2
```

А `Target{anchor_slot}` — двумя строками:

```text
t1 <target_if_up>
t2 <target_if_down>
```

Где `<target_if_up>` вычисляется как будто entry = `<last_close> + 2`, `<target_if_down>` — как будто entry = `<last_close> - 2`, с тем же правилом **min 10 pt** до ближайшего key level (иначе граница Expected Range).

Остальная цепочка слотов после `anchor_slot` продолжается по прежней логике `chained_entry_from_previous_target(prev_target ± 2)` и `intraday_target_from_entry`.

**Target:** `intraday_target_from_entry` — ближайший Key level ≥ **10 pt** от entry (оба направления в ER), иначе граница Expected Range; reverse после выхода за ER / Max Target.

**Max Target:** только bar-fact (экстремумы сегментов), не перезаписывается моделью.

### 8.2.3 Акцент на активном сценарии (визуальный режим)

- `Bias Trend` и `Chop` заполняются **только** в активной строке;
- во второй строке эти колонки становятся `"-"`;
- активная строка подсвечивается зелёным (ячейки колонок `Scenario` и `Bias Trend`).

`Key levels` сохраняются из source xlsx без пересчёта.

### 8.3 Chop/Bias только в активной строке

В текущей реализации update (операторский режим) `Chop` и `Bias Trend` переносятся в активную строку, а неактивная строка получает `"-"`.

### 8.4 Legacy layers (в коде, не в основном пути `run_update`)

Присутствуют, но **не вызываются** из `run_update()` напрямую:

- `apply_intraday_session_memory_layer` (B);
- `apply_intraday_segment_target_layer` (B′);
- `apply_intraday_session_path_layer` (C).

Основной intraday-путь для Entry/Target — **adaptive layer** (§8.2).

---

## 9. Запись Excel

1. Определение source xlsx:
   - если есть `update_TradePlan_<дата>_<i>.xlsx` → берём самый новый (макс. `i`);
   - иначе → `SPX_TradePlan_<дата>.xlsx`.
2. `shutil.copy2(source, update_TradePlan_<дата>_<i+1>.xlsx)`.
2. **Summary**: Open (actual), Chop duration (многострочный по сегментам), staged bias texts, EM / Expected Range из rebuild.
3. **Trade Plan**:
   - при успешном rebuild — `apply_trade_refresh_rows_to_trade_plan_sheet` (миграция legacy-колонок Entry4/Target4, Bar sequence);
   - иначе — только Chop + Bias Trend;
   - `apply_trade_plan_column_fills`.

Миграции листа: `_migrate_trade_plan_entry4_target4`, `_migrate_trade_plan_bar_columns`.

---

## 10. Зависимости модулей

| Модуль | Использование |
|--------|----------------|
| `bias` | таблица, staged Ridge, segment targets, combined audit |
| `chop` | predict, blended actuals, A2, reapply, regression cache |
| `build_TradePlan` | `assemble_trade_plan`, entry/target helpers, headers, fills |
| `levels` | не используется для пересчёта `Key levels` в update (колонка берётся из source xlsx) |
| `fresh_data_loader` | IBKR fetch last_data |
| `openpyxl` | чтение утреннего плана, запись update xlsx |

---

## 11. Ключевые функции

| Функция | Назначение |
|---------|------------|
| `run_update` | Главный orchestrator |
| `_detect_phase` | stage2 / stage3 / stage4 по времени бара |
| `_merge_session_bars` | Merge last_data в 90d CSV |
| `_predict_stage` | Bias Ridge на стадии |
| `rebuild_trade_plan_intraday` | assemble + staged bias texts |
| `apply_intraday_scenario_adaptive_layer` | Entry2–4 active scenario |
| `load_morning_trade_plan_rows` | Snapshot утренних Target/Entry |
| `segment_close_extremes_rth` | max/min Close по сегментам |
| `segment_last_close_rth` | Close последнего бара сегмента |
| `_chop_duration_summary_cell` | Текст колонки Chop |

Chop-функции — в [`chop.py`](../chop.py), см. [chop_algorithm.md §22](chop_algorithm.md#22-intraday-chop-update-path).

---

## 12. Типичный audit (`bias_last_prediction_debug.json`)

Поля, специфичные для update:

- `run_mode`: `intraday_update_stage2` / `stage3` / `stage4`;
- `intraday_phase`, `last_bar_time_et`, `actual_open_spx_last_data`;
- `bias_dir_*`, `bias_strength_*`, `bias_combined_*`, `bias_for_chop`;
- `chop_intraday_segment_actuals_frac`, `chop_intraday_actuals_anchored`, `chop_full_chain_no_anchor`;
- `predicted_chop_pct`, `predicted_chop_pct_1..4`;
- `effective_intraday_chop_pct`, `chop_guardrail`, `realized_mean_completed_chop_pct`;
- `intraday_pool_anchors`, `intraday_chop_reapply`;
- `trade_plan_intraday_rebuild`, `intraday_adaptive_layer`, `intraday_adaptive_layer_status`;
- `segment_close_extremes_rth`.

---

## 13. Отличия от утреннего `build_TradePlan`

| Аспект | Утро (`build_TradePlan`) | Update (`update_TradePlan`) |
|--------|--------------------------|-----------------------------|
| Open | `open_price.predict_spx_open` | **факт** Open из last_data |
| Bias/chop | полный прогноз 1–4 | факт + прогноз остатка |
| Chop pool | без session_range anchors | anchors + guardrail + reapply |
| Entry2–4 | утренние правила assemble | **adaptive** от last_close + KL |
| Target1 | assemble | утро **freeze** + контроль сегмента |
| Выход | `SPX_TradePlan_*` | `update_TradePlan_<дата>_<i>.xlsx` (версионирование) |

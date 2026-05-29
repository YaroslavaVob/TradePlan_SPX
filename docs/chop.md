# Chop algorithm specification (`chop.py`)

Документ фиксирует текущую реализацию [`chop.py`](../chop.py). Цель документа — описать модуль так, чтобы логика была понятна без чтения кода: какие данные нужны, как строится `chop_dur`, какие признаки считаются, как обучается Ridge, как работает калибровка, как выбирается набор признаков, как считаются EM/targets, как строятся chop-зоны и как по similarity-пулу считается **`sequence+range`** (пробеги знака C−O и HL-диапазоны).

Если описание расходится с кодом, источник истины — `chop.py`.

## 1. Назначение модуля

`chop.py` прогнозирует для следующей SPX RTH-сессии:

- `predicted_chop_pct`: ожидаемая доля chop/flat минут в RTH, в процентах (**стадированный Ridge по сегментам**, [§5.1](#51-staged-chop-at-inference));
- при наличии инференса по сегментам: `predicted_chop_pct_1`…`_4` (доли по четырём интервалам RTH);
- `mean_range_spx_bull` и `mean_range_spx_bear`: ожидаемый диапазон SPX отдельно для bullish/bearish сценариев;
- scenario targets: `target_bullish_high/low`, `target_bearish_high/low`;
- `chop_zones`, `bullish_chop_zones`, `bearish_chop_zones`: исторические price zones, где рынок часто находился во flat/chop-сегментах;
- **`sequence+range`**: сводка по длинам и High–Low диапазонам «пробегов» знака тела свечи (C−O) на похожих исторических сессиях, отдельно для весов bull и bear ([§17](#17-sequencerange-co-sign-runs-and-hl-range-on-similar-sessions));
- диагностические поля для EM, targets, model metrics и использованных features.

Основная шкала модели:

- target в модели: `chop_dur` в диапазоне `[0, 1]`;
- пользовательский вывод: `predicted_chop_pct = chop_dur * 100`.

## 2. Текущая зафиксированная версия

Ключевые параметры текущей версии (сверяйте с `chop.py`):

- `RUN_OFFLINE_SELECTION` — в репозитории по умолчанию **`True`** (полный офлайн/research: пересборка таблицы, Stage 1 отбора признаков и т.д.); для ежедневного режима без полного offline selection выставляют **`False`**;
- `CHOP_OFFLINE_INCLUDE_CATBOOST = False` — CatBoost-ветка выключена; при **`True`** и **`RUN_OFFLINE_SELECTION=True`** дополнительно запускаются эксперимент CatBoost и сохранение вторичного артефакта;
- устаревший `CHOP_RUN_MODE` закомментирован; источник режима — только `RUN_OFFLINE_SELECTION`;
- основная Ridge-калибровка: `CHOP_PRIMARY_CALIBRATION_MODE = "scale"`;
- режим калибровки CatBoost: `CHOP_CATBOOST_CALIBRATION_MODE = "quantile"` — см. [§12](#12-catboost-experiment-and-secondary-model);
- сетка `alpha` для `GridSearchCV`: `RIDGE_ALPHA_GRID = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]`;
- `TIME_SERIES_SPLITS = 5`;
- holdout для офлайна и метрик отбора: `N_TEST_SESSIONS = 27` (последние **27** целевых дат — тест);
- верхняя граница строк для подбора гиперпараметров: `N_SESSIONS_FOR_HP = 80` (первые по времени сеансы таблицы);
- **Stage 1 (Ridge и CatBoost pruning)**: фиксированной константы «число итераций» нет; используются **`_chop_loo_removal_steps(n_start, MIN_FEATURES_AFTER_SELECTION)`** (= сколько раз удаляется один признак) и **`_chop_feature_selection_total_runs`** (= это число **+ 1** финальный прогон на нижней границе без дальнейшего дропа). В лог печатается и число leave-one-out удалений `(n_start→MIN)`, и число прогонов GridSearch;
- минимум признаков после отбора: `MIN_FEATURES_AFTER_SELECTION = 14`;
- в `CHOP_FEATURE_ORDER` **36** имён столбцов таблицы (порядок — в `chop.py`; добавлен **`session_range_gap`**, см. [§6.26](#626-session_range_gap-ridge-yes));
- дополнительно в таблице (не в `CHOP_FEATURE_ORDER`, но в кэше): **`session_range_1`…`session_range_4`** — SPX pts по RTH-сегментам; **исключены из Ridge**, используются в intraday similarity ([§14.2](#142-intraday-session_range-anchors-and-post-filter), [§22](#22-intraday-chop-update-path));
- признаки Ridge/CatBoost: пересечение с регрессионной таблицей **минус** `CHOP_FEATURES_EXCLUDED_FROM_MODEL` ([§6](#6-feature-set-formulas-and-calculation-logic));
- финальный бленд 30m/20m таргета по сегменту: `CHOP_FINAL_SHORT_BLEND_W = 0.80` (притяжение короткого горизонта к **медиане** `chop_dur_20_k` по истории — [§4.3](#43-segment-targets-session-chop_dur-and-3020-columns));
- **якорение EM** к референсному диапазону: `CHOP_EM_ANCHOR_W_POOL = 0.30`, `CHOP_EM_ANCHOR_W_RECENT = 0.70`, `CHOP_EM_ANCHOR_BETA = 0.60` ([§15](#15-em-calculation));
- **мягкое якорение** `targets_range_scale` к тому же референсу: `CHOP_TARGETS_ANCHOR_BETA = 0.20` ([§16](#16-scenario-targets));
- **intraday chop guardrail** (сильный chop day по первым сегментам): `INTRADAY_CHOP_GUARDRAIL_MEAN_HIGH = 0.70` → floor **0.55**; `INTRADAY_CHOP_GUARDRAIL_MEAN_MED = 0.60` → floor **0.45** ([§22.2](#222-layer-a2-effective-chop-and-guardrail));
- intraday similarity по **`session_range_k`**: вес `CHOP_SIM_W_INTRADAY_RANGE_SEG = 0.30`, post-filter band **1.25 → 1.40** ([§14.2](#142-intraday-session_range-anchors-and-post-filter)).

Финальный выбор **лучшего прогона** Stage 1 между итерациями (после всех удалений признаков):

`MonoBuckets → Spearman → DirAcc → BucketPenalty → MAE`

где **BucketPenalty** = \(\sum_b \max(0,\;\overline{y}_{b-1}-\overline{y}_b)\) по средним фактическим `chop_dur` в **6** корзинах, упорядоченных по предсказанию (меньше штраф — лучше). См. `_pick_best_chop_run_index`.

На каждой итерации **leave-one-feature-out** используется **тот же** порядок ранжирования на holdout (минимизация кортежа `(-Mono,-Spear,-DirAcc, BucketPenalty, MAE)`).

## 3. Data Sources And Calendar

Data directory:

- `DATA_DIR = data_spx`

Input files loaded by `_load_chop_data()`:

- `SPX_1min_90d_ibkr.csv`: SPX 1m, main calendar, RTH features, target `chop_dur`, actual high/low/range, zones;
- `ES_1min_90d.csv`: ES 1m, premarket 00:00-09:30 for D and ES overnight window for zone logic;
- `ES_1min_90d_RTH.csv`: ES RTH 1m for `vwap_reversion_gap`;
- `VIX_5min_90d.csv`: VIX 5m for IV features;
- `VIX9D_5min_90d.csv`: VIX9D 5m for short/long IV spread;
- `SPX_1d_1y.csv`: SPX daily OHLC, loaded through `load_spx_1d()` for daily ATR and close-path features.

Datetime handling:

- `_ensure_datetime_tz()` converts timezone-aware timestamps to NY timezone and then removes tz;
- `_ensure_dt()` adds `Date = Datetime.dt.date` and `Time = Datetime.dt.time`;
- RTH mask is `[09:30, 15:59]`;
- VIX RTH window is `[09:30, 16:00)`;
- ES premarket / overnight window is `[00:00, 09:30)`.

Trading dates:

- `_get_trading_dates_from_1m()` takes unique SPX RTH dates from `SPX_1min_90d_ibkr.csv`;
- `D`, `D-1`, ..., `D-4` are positions in this SPX session list, not calendar-day arithmetic;
- first possible target D is index `4`, because features require D-1..D-4;
- `_first_target_date_with_es_premarket()` additionally requires ES premarket for target D.

## 4. Target Variable `chop_dur`

`chop_dur(D)` is the fraction of SPX RTH minutes covered by at least one flat 30-minute window:

```text
chop_dur(D) = covered_minutes(D) / 390
```

The implementation path:

- `calculate_actual_chop_dur()`
- `_chop_dur_flat_window_coverage()`
- `_flat_windows_pass_filters()`

### 4.1 ATR(14) For 1m Bars

`_atr14_wilder_1m_array(high, low, close)`:

1. True Range starts from bar 2:
   `TR_i = max(H_i-L_i, abs(H_i-C_{i-1}), abs(L_i-C_{i-1}))`.
2. Initial ATR is the mean of `TR_1..TR_14`.
3. Wilder smoothing:
   `ATR_i = (ATR_{i-1} * 13 + TR_i) / 14`.
4. Values before the first valid ATR are `NaN`.

Constants:

- `ATR_INIT_BARS = 15`;
- first valid ATR is around bar 15/16 depending on zero-based indexing.

### 4.2 Flat Window Filters

Rolling window:

- length = `FLAT_WINDOW_MINUTES = 30`;
- source = SPX RTH 1m bars.

A 30-minute window passes only if all filters pass:

1. Range filter:

```text
window_range <= K1_FLAT_ATR * median(ATR_window)
```

where `K1_FLAT_ATR = 5.0`.

2. Trend/drift filter:

```text
abs(Close_last - Close_first) <= K2_TREND_BLOCK * window_range
```

where `K2_TREND_BLOCK = 0.35`.

3. Price density filter:

```text
q90(Close_window) - q10(Close_window) <= K3_PRICE_DENSITY * median(ATR_window)
```

where `K3_PRICE_DENSITY = 3.5`.

The target does not count windows. It unions all minutes covered by passing windows:

- overlapping windows are counted once;
- `covered_minutes / RTH_MINUTES` gives `chop_dur`;
- output is clipped to `[0, 1]` when stored in the table.

Important distinction:

- target `chop_dur`: union of all passed flat-window minutes;
- feature `chop_duration_gap`: uses merged flat segments from the zone pipeline, not exactly the same coverage calculation.

### 4.3 Segment targets, session `chop_dur`, and 30/20 columns

Помимо сессионного таргета (объединение flat-окон на всём RTH), таблица хранит **четыре RTH-сегмента** в той же сетке, что и `bias.py` (`BIAS_SEGMENT_*_MINUTES`, сумма 390 минут):

- для каждого `k ∈ {1,2,3,4}`: `chop_dur_30_k`, `chop_dur_20_k` — доля минут сегмента, покрытая объединением проходящих flat-окон длины 30 или 20 минут (`_chop_segment_flat_coverage_fraction`, те же фильтры K1/K2/K3);
- столбцы `chop_dur_30`, `chop_dur_20` на строке D — те же метрики, но на **полной** RTH-сессии (`calculate_actual_chop_dur` с `window_minutes=30` или `20`); они сохраняются в таблице и отделены от `chop_dur_*_k`.

После сборки строк `finalize_chop_segment_and_session_targets_inplace()`:

1. По **всему датафрейму** берётся медиана столбца `chop_dur_20_k` (`med_k`).
2. Для каждой строки смешивается короткий и длинный горизонт сегмента:

```text
chop_dur_k = clip(chop_dur_30_k + CHOP_FINAL_SHORT_BLEND_W * (chop_dur_20_k - med_k), 0, 1)
```

   (идентично `blend_chop30_with_chop20` для инференса и intraday-факта — `chop_intraday_blended_actuals_from_trimmed_session_block`, [§22.1](#221-blended-segment-actuals-from-live-bars).)

**Притяжение к медиане (median boundary pull):** член `(chop_dur_20_k - med_k)` смещает смешанный таргет только на **отклонение** 20m-окна от исторической медианы по столбцу `chop_dur_20_k` (медиана по **всей** регрессионной таблице на этапе `finalize_*`; на intraday для факта сегмента — медиана того же столбца из кэша). Если `chop_dur_20_k ≈ med_k`, вклад 20m **обнуляется** и остаётся `chop_dur_30_k`; экстремальные 20m-значения притягиваются к «типичному» уровню с весом `CHOP_FINAL_SHORT_BLEND_W`.

3. Сессионный `chop_dur` перезаписывается как минутно-взвешенная сумма сегментов:

```text
chop_dur = (chop_dur_1*m1 + chop_dur_2*m2 + chop_dur_3*m3 + chop_dur_4*m4) / RTH_MINUTES
```

   (`session_weighted_chop_from_segments`).

Офлайн Ridge по **сессии** и отбор признаков в Stage 1 используют именно этот агрегированный `chop_dur` как `y`. Сегментные столбцы `chop_dur_*_k` и смешанные `chop_dur_k` входят в `CHOP_FEATURES_EXCLUDED_FROM_MODEL` и в Ridge не подаются.

## 5. Regression Table

`build_chop_table()` returns:

```text
(regression_df, train_dates, test_dates)
```

Each row is one target date `D`.

Row contents (помимо всех полей из `build_chop_features`):

- `date`: целевая сессия D;
- сегментные таргеты `chop_dur_30_k`, `chop_dur_20_k` для `k=1..4`;
- сессионные `chop_dur_30`, `chop_dur_20` (full RTH, до финализации);
- после `finalize_chop_segment_and_session_targets_inplace`: `chop_dur_k` и итоговый **`chop_dur`** (взвешенный по минутам);
- `actual_range_spx`, `actual_high_spx`, `actual_low_spx` с RTH D;
- **`bias_dir`**, **`bias_strength`**: для исторических строк — фактический signed bias целевого дня D из `_session_bias_dir_strength_row` / `_y_bias_direction_strength_session` (полный RTH D + дневной ATR из дневного ряда);
- **`bias_combined`**: добавляется в конце через `_chop_add_bias_combined_column` — та же логика, что в `bias.py`: эмпирические квантильные пороги по `bias_dir` и `bias_strength`, затем `bias_segment_combined_from_raw(...)`.

Наследие: в `_pool_bias_strength_series()` при отсутствии новых колонок возможны откаты на `y_bias` или `bias`, если они есть в таблице; текущее построение таблицы через `build_chop_table` задаёт именно **`bias_dir` / `bias_strength`**.

Row construction sequence:

1. Построить список торговых дат SPX; первый D — индекс `FIRST_TARGET_SESSION_INDEX = 4` (нужны D-1…D-4).
2. `first_target` — первая дата с доступным ES premarket для D (`_first_target_date_with_es_premarket`).
3. Для каждой подходящей D (и опционального исключения `exclude_predict_session_date` — строка `--plan-for` из `main` не попадает в таблицу):
   - признаки: `build_chop_features(..., D, ...)` **без** `exclude_spx_rth_on_date` (полный SPX для истории D);
   - таргеты сегментов и сессии — как в [§4.3](#43-segment-targets-session-chop_dur-and-3020-columns);
   - `finalize_chop_segment_and_session_targets_inplace`, затем `_chop_add_bias_combined_column`.
4. Никакой глобальной интерполяции/заполнения по всей таблице на этапе сборки.
5. Разбиение при `build_chop_table(..., n_test_sessions=N_TEST_SESSIONS)`:
   - последние `N_TEST_SESSIONS` целевых дат → `test_dates`;
   - остальные → `train_dates`.

Инкрементальный режим (`update_chop_regression_table_cache_incremental` в `chop.py`) работает с файлом кэша `chop_regression_table.csv.gz`:

- **Первичная сборка**: если кэша нет, он пустой или в нём нет колонки `date` — один раз вызывается **`build_chop_table`** (полная таблица признаков и таргетов по истории SPX).
- **Миграция старого кэша**: если в сохранённом кэше **нет** колонок `chop_dur_30`, `chop_dur_20` или `chop_dur_30_1` (устаревший формат), **или** нет **`session_range_1`** — снова вызывается **`build_chop_table`** и кэш **полностью заменяется**.
- **Обычное обновление**: иначе к уже сохранённым датам **добавляются только отсутствующие** торговые дни, затем `finalize_chop_segment_and_session_targets_inplace` и **`_chop_add_bias_combined_column`**. Если новых дат нет, вызываются **finalize** и при необходимости дозаполнение **`bias_combined`**.

Инференс и поиск похожих сессий опираются на **строки этой исторической таблицы** (кэш + догрузка); отдельной ветки «пересобрать всё ради фильтра» нет.

Кэш: `_save_chop_regression_table_cache()` → `chop_regression_table.csv.gz`; `exclude_predict_session_date` вычищается из кэша при обновлении.

## 5.1 Staged chop at inference

Главное число **`predicted_chop_pct`** в `predict_session_chop()` строится **не** через единственный сохранённый `chop_regression_model.json` на сессию, а через стадированный блок `_chop_staged_predict_session_chop`:

1. Для каждого сегмента `k`:
   - базовые признаки Ridge: `_chop_base_model_feature_cols(regression_df)` — порядок `CHOP_FEATURE_ORDER` минус исключения, или `selected_features` из JSON, если схема валидна (`_chop_hyperparams_valid`);
   - добавляются **дополнительные** колонки по стадии: `CHOP_SEGMENT_STAGE_EXTRA_FEATURES` — `{}` для k=1, `{chop_dur_1}` для k=2, `{chop_dur_1,chop_dur_2}` для k=3, `{chop_dur_1,chop_dur_2,chop_dur_3}` для k=4;
   - обучаются **две** временные модели Ridge на **всей** `regression_df`: таргеты `chop_dur_30_k` и `chop_dur_20_k` (`train_chop_model(..., persist_to_disk=False)`);
   - предсказания блендятся: медиана исторического столбца `chop_dur_20_k` по таблице + `blend_chop30_with_chop20`; результат записывается в словарь признаков как `chop_dur_k`;
2. Сессионная доля chop: `session_weighted_chop_from_segments(chop_dur_1..4)` — то же взвешивание минутами, что в [§4.3](#43-segment-targets-session-chop_dur-and-3020-columns);
3. **`predict_chop_ratio` / сохранённый JSON** используются для исторической валидации одной модели на таргете `chop_dur`, CatBoost-сравнения и утилит; дневной «главный» процент chop — из стадированного пути выше.

**Intraday** (`intraday_predict_stages`, `intraday_chop_segment_actuals`, `intraday_chop_preserved_frac`): для стадий вне набора предсказания подставляются зафиксированные доли из факта или сохранённых долей; в `stage_details` помечается `mode`: `predicted`, `fixed`, `missing`.

Консольный вывод: `_print_chop_staged_inference_console` (вес сессии и по сегментам).

## 6. Feature Set: formulas and calculation logic

`CHOP_FEATURE_ORDER` задаёт **36** признаков таблицы. Все они попадают в CSV/cache, но столбцы из `CHOP_FEATURES_EXCLUDED_FROM_MODEL` **никогда** не подаются в Ridge или CatBoost-модель (и отфильтровываются, если встретились в `selected_features`).

Исключённые из модели (**таблица / диагностика / EM/targets / similarity**):

- из «логики bias / IV / уровня / премаркета»: `bias_strength`, `bias_dir`, `atr_level_ratio`, `bar_compression_gap`, `pm_chop_potential`, **`pm_iv_rv_dislocation`**, **`vix_ratio_spx`** (`vix_ratio_spx` и **`pm_iv_rv_dislocation`** остаются в таблице для диагностики и сопоставления с историей в similarity);
- все сегментные таргеты: `chop_dur_1`…`chop_dur_4`, `chop_dur_30_1`…`chop_dur_30_4`, `chop_dur_20_1`…`chop_dur_20_4`;
- **`session_range_1`…`session_range_4`**: High−Low SPX pts по четырём RTH-сегментам (те же срезы, что bias/chop staged); в Ridge не входят, нужны для intraday similarity и post-filter ([§14.2](#142-intraday-session_range-anchors-and-post-filter)).

При полном наборе колонок после исключений остаётся до **19** кандидатов в Ridge; точный список в JSON — `selected_features`.

Эти исключённые столбцы нужны для staged-инференса, EM, targets, зон и диагностики.

Notation:

- target row date = `D`;
- historical sessions = `D-1`, `D-2`, `D-3`, `D-4`;
- `mean_prev(x)` = mean of finite values over available `D-2..D-4`;
- `eps = CANDLE_RATIO_EPS = 1e-6`;
- `log_ratio(a,b) = log((a + eps) / (b + eps))`;
- SPX intraday features use SPX 1m RTH unless explicitly stated otherwise;
- ES premarket features use ES 1m `[00:00, 09:30)` on target date `D`.

### 6.1 `chop_ratio_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. For session `d`, compute ATR(14) on 1m bars.
2. Slide 30-minute windows through RTH.
3. Count windows passing the same K1/K2/K3 flat filters used by chop zones:
   - `window_range <= K1_FLAT_ATR * median(ATR_window)`;
   - `abs(Close_last - Close_first) <= K2_TREND_BLOCK * window_range`;
   - `q90(Close_window) - q10(Close_window) <= K3_PRICE_DENSITY * median(ATR_window)`.
4. `atr_flat_share(d) = passed_windows / total_rolling_windows`.

Feature:

```text
chop_ratio_gap =
  atr_flat_share(D-1)
  - mean(atr_flat_share(D-2), atr_flat_share(D-3), atr_flat_share(D-4))
```

The value is rounded to 3 decimals in code.

### 6.2 `chop_duration_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Build flat segments using `_flat_segments_spx_rth_day()`.
2. The segment pipeline:
   - find 30-minute flat windows by K1/K2/K3 filters;
   - merge windows if the gap is `<= MERGE_GAP_MINUTES`;
   - drop merged segments shorter than `MIN_ZONE_DURATION_MIN`;
   - keep only segments with `inside_share >= INSIDE_SHARE_MIN`.
3. `atr_chop_dur(d) = sum(segment.duration_min) / RTH_MINUTES`.

Feature:

```text
recent = mean(atr_chop_dur(D-1), atr_chop_dur(D-2))
older  = mean(atr_chop_dur(D-3), atr_chop_dur(D-4))
chop_duration_gap = recent - older
```

This differs from target `chop_dur`, which uses union coverage of all passing windows.

### 6.3 `range_ratio_spx` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. For each rolling 30-minute window:
   - `window_range = max(High_window) - min(Low_window)`;
   - `atr_w = median(ATR_window)`.
2. Keep finite windows with positive ATR.
3. `range_norm30(d) = median(window_range / atr_w)`.

Feature:

```text
range_ratio_spx =
  log_ratio(range_norm30(D-1), mean_prev(range_norm30))
```

### 6.4 `pm_range_norm` (Ridge: yes)

Source: ES 1m premarket D + SPX daily ATR.

Premarket metric:

```text
pre_range(D) = max(High_premarket_D) - min(Low_premarket_D)
```

Daily ATR:

```text
atr_d1 = ATR14_daily(D-1)
```

Feature:

```text
pm_range_norm = pre_range(D) / (atr_d1 + eps)
```

### 6.5 `pm_directional_eff` (Ridge: yes)

Source: ES 1m premarket D.

Feature:

```text
pm_directional_eff =
  abs(Close_last_premarket - Close_first_premarket)
  / (pre_range(D) + eps)
```

Interpretation:

- close to 1: directional premarket;
- close to 0: two-sided/choppy premarket.

### 6.6 `pm_chop_potential` (Ridge: no)

Source: derived from `pm_range_norm` and `pm_directional_eff`.

Feature:

```text
pm_chop_potential =
  pm_range_norm * (1 - pm_directional_eff)
```

Interpretation:

- large premarket range with low directional efficiency increases chop potential;
- исключён из Ridge/CatBoost; расстояние similarity для пула EM/targets строится по **`pm_range_norm`**, **`pm_iv_rv_dislocation`** и **`pm_vwap_stickiness`** ([§14](#14-similarity-pool-for-em-and-targets)), а не по этому столбцу.

### 6.7 `pm_vwap_stickiness` (Ridge: yes)

Source: ES 1m premarket D.

Session metric:

1. Compute VWAP over premarket:

```text
TP = (High + Low + Close) / 3
VWAP_i = cumulative_sum(TP_i * Volume_i) / cumulative_sum(Volume_i)
```

If volume is missing/invalid, fallback volume is `1.0`.

2. Compute normalized distance:

```text
dist_norm_i = abs(Close_i - VWAP_i) / (pre_range(D) + eps)
```

Feature:

```text
pm_vwap_stickiness = 1 - median(dist_norm_i)
```

Higher value means premarket stayed closer to VWAP.

### 6.7a `pm_iv_rv_dislocation` (Ridge: no)

Источники: VIX 5m и ES 1m в **премаркете** целевого дня `D` (`Datetime.dt.time < 09:30`).

```text
pm_iv_front(D) = median(VIX Close в премаркете D) / 100 / sqrt(252)
pm_rv_proxy(D) = median(rolling_30m_range_ES_premarket(D)) / ES_premarket_last_close(D)
pm_iv_rv_dislocation(D) = log((pm_iv_front + eps) / (pm_rv_proxy + eps))
```

Столбец **исключён из Ridge/CatBoost**, но хранится в регрессионной таблице и используется в **`_weighted_similarity_pool_for_targets`**: сравнение премаркетного IV/RV прогнозной сессии с историческими строками (`pred_pm_iv_rv_dislocation` vs колонка `pm_iv_rv_dislocation`).

### 6.8 `bollinger_compression_ratio` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Bollinger settings:
   - period = `BOLLINGER_PERIOD = 20`;
   - k = `BOLLINGER_K = 2.0`.
2. For each bar after warmup:

```text
SMA = rolling_mean(Close, 20)
STD = rolling_std(Close, 20)
upper = SMA + 2 * STD
lower = SMA - 2 * STD
bbw_ratio = (upper - lower) / Close
```

3. `bbw_abs(d) = median(bbw_ratio after warmup)`.

Feature:

```text
bollinger_compression_ratio =
  log_ratio(bbw_abs(D-1), mean_prev(bbw_abs))
```

### 6.9 `bias_dir` и `bias_strength` (Ridge: no)

Источник:

- **Историческая строка таблицы** (обучение): SPX 1m полного RTH целевого дня D и дневной ATR через `_session_bias_dir_strength_row` → `_y_bias_direction_strength_session` (аналог направленности/силы bias в `bias.py`, не скалярное «тело свечи по минуткам» как старый `y_bias`).
- **Инференс** для даты `target_date`: SPX RTH на D **вырезан** из потока (`exclude_spx_rth_on_date=target_date` в `build_chop_features`), поэтому признаки считаются без утечки intraday целевого дня; **`bias_dir` / `bias_strength` подставляются из прогноза bias** (`predict_bias_and_probability` → `bias_dir_4`, `bias_strength_4`) или из `session_bias_override` / `bias_for_chop_combined` в `predict_session_chop`.

Колонки **исключены из Ridge**: это сигнал дня D; для таблицы они отражают факт, для прогноза — внешнюю модель bias, а не Ridge-признаки.

На таблице после сборки добавляется **`bias_combined`** ([§5](#5-regression-table)).

### 6.10 `vix_ratio_spx` (Ridge: no)

Source: VIX 5m, VIX9D 5m, SPX 1m RTH.

Session IV metric:

1. For each session `d`, take VIX RTH bars `[09:30, 16:00)`.
2. Compute:
   - `m_vix = median(VIX Close)`;
   - `m_9d = median(VIX9D Close)`.
3. If both exist:

```text
combined_iv(d) = (m_vix + m_9d) / 2
```

Otherwise use the available one.

SPX normalization:

```text
range30_med(d) = median(max(High_30m_window) - min(Low_30m_window))
vix_adj(d) = combined_iv(d) / (range30_med(d) + eps)
```

Feature:

```text
vix_ratio_spx = log_ratio(vix_adj(D-1), mean_prev(vix_adj))
```

**Исключён из Ridge/CatBoost**, но остаётся в таблице для диагностики и обратной совместимости.

### 6.10a Дополнительные gap-признаки (Ridge: yes, кроме явных исключений)

Ниже признаки строятся в `build_chop_features`; общий шаблон для «`*_gap`» от **D-1** относительно среднего по доступным **D-2…D-4** (ровно как в коде: см. `_feature_*_gap`).

**`compression_velocity_gap`** (SPX 1m RTH):

- `bbw_abs(d) = median((Upper-Lower)/Close)` по Боллинджеру на сессии `d`;
- `compression_velocity(d) = bbw_abs(d) - bbw_abs(prev_session)`;
- `compression_velocity_gap = compression_velocity(D-1) - mean(compression_velocity(D-2..D-4))`.

**`rotation_gap`** (SPX RTH):

- `ret_i = Close_i - Close_{i-1}`, знаки нулевых доходностей в доле чередований не входят;
- `rotation(d) = mean(sign(ret_i) != sign(ret_{i-1}))` по валидным парам;
- `rotation_gap = rotation(D-1) - mean(rotation(D-2..D-4))`.

**`vwap_cross_density_gap`** (SPX RTH): VWAP по типичной цене и объёму; `dist_i = Close_i - VWAP_i`; доля смен знака `dist` между соседними барами по валидным шагам; gap от **D-1** к среднему **D-2…D-4**.

**`range_efficiency_nonlinear_gap`** (SPX RTH, окно **30m**):

- для каждого окна `eff_w = abs(Close_end - Close_start) / (High_w - Low_w + eps)`;
- `range_efficiency_nonlinear(d) = 1 - median(eff_w)`;
- gap от **D-1** к среднему **D-2…D-4**.

**`return_entropy_gap`** (SPX RTH):

- `ret_i` как выше; `q = median(abs(ret_i))` внутри сессии; три корзины `-1/0/+1` по порогам `±q`; энтропия Шеннона по долям корзин;
- gap от **D-1** к среднему **D-2…D-4**.

**`failed_expansion_streak_gap`** (SPX RTH):

- «failed event»: пробой за пределы prior **30m** бокса и возврат внутрь в течение **10m** (`FAILED_BREAKOUT_WINDOW_MINUTES`, `FAILED_BREAKOUT_RETURN_MINUTES`);
- по сессии — максимум длины подряд идущих таких событий;
- gap от **D-1** к среднему **D-2…D-4**.

**`iv_skew_change_gap`** (VIX / VIX9D RTH, поканально где возможно):

- `iv_skew(d) = median(VIX9D - VIX)` по объединённым барам;
- `iv_skew_change_gap = (iv_skew(D-1)-iv_skew(D-2)) - mean(iv_skew(D-2)-iv_skew(D-3), iv_skew(D-3)-iv_skew(D-4))` (реализация собирает доступные одношаговые дельты по **D-2…D-4**).

**`iv_rv_dislocation_gap`** (SPX RTH + VIX RTH):

- `rv_intraday(d) = median(rolling_30m_range(d)) / Close_d`;
- `iv_front(d) = median(VIX9D RTH)/100/sqrt(252)`, fallback `median(VIX RTH)/100/sqrt(252)`;
- `iv_rv_dislocation(d) = log((iv_front + eps)/(rv_intraday + eps))`;
- gap от **D-1** к среднему **D-2…D-4**.

### 6.11 `bar_compression_gap` (Ridge: no)

Source: SPX 1m RTH.

Session metric:

```text
median_range(d) = median(High_i - Low_i)
```

Historical scale:

```text
hist_mean = mean(median_range(D-2), median_range(D-3), median_range(D-4))
```

Normalize:

```text
cur_score = median_range(D-1) / (hist_mean + eps)
prev_score(d) = median_range(d) / (hist_mean + eps)
mean_prev_score = mean(prev_score(D-2..D-4))
```

Feature:

```text
bar_compression_gap = log_ratio(cur_score, mean_prev_score)
```

Excluded from Ridge.

### 6.12 `candle_iqr_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

```text
body_abs_i = abs(Close_i - Open_i)
bar_range_i = High_i - Low_i
candle_iqr_ratio(d) =
  IQR(body_abs_i) / (median(bar_range_i) + eps)
```

Feature:

```text
candle_iqr_gap =
  mean(candle_iqr_ratio(D-2), candle_iqr_ratio(D-3), candle_iqr_ratio(D-4))
  - candle_iqr_ratio(D-1)
```

The sign is intentionally previous mean minus D-1.

### 6.13 `alternation_inside_box_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Compute every rolling 30-minute window range.
2. Compute `q40` of those ranges.
3. Keep only windows where:

```text
box_range <= q40(session_30m_ranges)
```

4. Inside each kept box:
   - compute candle body `Close - Open`;
   - take signs;
   - remove zero signs;
   - compute share of adjacent sign changes.
5. `alternation_inside_box(d)` = mean alternation share over kept narrow boxes.

Feature:

```text
alternation_inside_box_gap =
  alternation_inside_box(D-1)
  - mean_prev(alternation_inside_box)
```

### 6.14 `spx_vwap_stickiness_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Compute SPX RTH VWAP:

```text
TP = (High + Low + Close) / 3
VWAP_i = cumulative_sum(TP_i * Volume_i) / cumulative_sum(Volume_i)
```

2. Compute ATR(14) on SPX 1m.
3. For finite bars with valid ATR:

```text
ratio_i = abs(Close_i - VWAP_i) / (ATR_i + eps)
```

4. Session stickiness:

```text
vwap_stickiness(d) = 1 - median(ratio_i)
```

Feature:

```text
spx_vwap_stickiness_gap =
  vwap_stickiness(D-1) - mean_prev(vwap_stickiness)
```

### 6.15 `range_efficiency_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

For each rolling 30-minute window:

```text
window_range = max(High_window) - min(Low_window)
eff = abs(Close_last - Close_first) / (window_range + eps)
```

`low_eff_share(d)`:

```text
count(eff <= RANGE_EFF_LOW_THRESHOLD) / count(valid_windows)
```

where `RANGE_EFF_LOW_THRESHOLD = 0.25`.

Feature:

```text
range_efficiency_gap =
  low_eff_share(D-1) - mean_prev(low_eff_share)
```

### 6.16 `impulse_bar_share_gap` (Ridge: yes)

Source: SPX 1m RTH.

Historical scale:

```text
hist_scale = median(median_bar_range(D-2), median_bar_range(D-3), median_bar_range(D-4))
```

Session metric:

```text
impulse_share(d) =
  count((High_i - Low_i) > IMPULSE_BAR_K * hist_scale) / N_bars
```

where `IMPULSE_BAR_K = 2.5`.

Feature:

```text
impulse_bar_share_gap =
  impulse_share(D-1) - mean(impulse_share(D-2), impulse_share(D-3), impulse_share(D-4))
```

### 6.17 `close_5d_eff` (Ridge: yes)

Source: SPX daily.

Despite the name, current implementation uses the four-session path `D-4..D-1`.

Close path:

```text
closes = [Close(D-4), Close(D-3), Close(D-2), Close(D-1)]
close_path = abs(Close(D-3)-Close(D-4))
           + abs(Close(D-2)-Close(D-3))
           + abs(Close(D-1)-Close(D-2))
```

Feature:

```text
close_5d_eff =
  abs(Close(D-1) - Close(D-4)) / (close_path + eps)
```

Interpretation: directional efficiency of the recent daily close path.

### 6.18 `iv_overnight_gap` (Ridge: yes)

Source: VIX 5m. Вызов: `bias._iv_overnight_gap` (внутри используются робастные z/клипы из `bias`, например `_iv_zscore_from_prev`; ниже — упрощённая схема для чтения, не побитовое совпадение с каждым guard в `bias.py`).

Conceptual calculation:

1. Compute VIX overnight delta for target D:

```text
overnight_delta(D) =
  VIX_preopen_last(D) - VIX_RTH_close(previous_session)
```

2. Historical comparison set:

```text
hist = [
  overnight_delta(D-1, D-2),
  overnight_delta(D-2, D-3),
  overnight_delta(D-3, D-4)
]
```

3. Return a z-score style gap using finite historical values and std floor guards.

Feature:

```text
iv_overnight_gap =
  (overnight_delta(D, D-1) - mean(hist)) / (std(hist) + eps)
```

This is one of the current-context inputs for similarity.

### 6.19 `iv_short_long_spread` (Ridge: yes)

Source: VIX 5m + VIX9D 5m. Вызов: `bias._iv_short_long_spread`; детали усреднений/робастной шкалы — в `bias.py`.

Session metric:

```text
spread(d) = VIX9D_RTH_close(d) - VIX_RTH_close(d)
```

Feature:

```text
iv_short_long_spread =
  (spread(D-1) - mean(spread(D-2), spread(D-3), spread(D-4)))
  / (std(spread(D-2), spread(D-3), spread(D-4)) + eps)
```

It captures short-vs-long implied-volatility term structure pressure.

### 6.20 `closepos_ratio_bbw` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Build Bollinger bands with period 20 and k=2.
2. After warmup:

```text
close_pos_i = (Close_i - lower_i) / (upper_i - lower_i)
```

3. Count how often close is near the Bollinger midline:

```text
bb_mid_share(d) =
  share(abs(close_pos_i - 0.5) <= BB_MID_SHARE_WIDTH)
```

where `BB_MID_SHARE_WIDTH = 0.20`.

Feature:

```text
closepos_ratio_bbw =
  bb_mid_share(D-1) - mean_prev(bb_mid_share)
```

### 6.21 `atr_level_ratio` (Ridge: no)

Source: SPX 1m RTH + SPX daily ATR.

Session metric:

```text
intraday_atr_median(d) = median(ATR_1m_tail(d))
daily_atr(d) = ATR14_daily(d)
atr_level(d) = intraday_atr_median(d) / (daily_atr(d) + eps)
```

Feature:

```text
atr_level_ratio =
  log_ratio(atr_level(D-1), mean_prev(atr_level))
```

Excluded from Ridge.

### 6.22 `inside_cluster_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Compute session median 1m range.
2. For each bar after the first:
   - `is_inside = High_i <= High_{i-1} and Low_i >= Low_{i-1}`;
   - `is_small = range_i <= INSIDE_CLUSTER_SMALL_BAR_RATIO * median_range`.
3. A bar is marked if `is_inside or is_small`.
4. Consecutive marked bars form a cluster.
5. Keep clusters with length >= `INSIDE_CLUSTER_MIN_LEN`.
6. `inside_cluster_ratio(d) = minutes_in_kept_clusters / N_bars`.

Feature:

```text
inside_cluster_gap =
  inside_cluster_ratio(D-1) - mean_prev(inside_cluster_ratio)
```

### 6.23 `vwap_reversion_gap` (Ridge: yes)

Source: ES RTH 1m (`ES_1min_90d_RTH.csv`).

Session metric:

1. Build ES RTH VWAP:

```text
TP = (High + Low + Close) / 3
VWAP_i = cumulative_sum(TP_i * Volume_i) / cumulative_sum(Volume_i)
```

2. Compute ES ATR(14) on 1m bars.
3. For finite bars:

```text
norm_i = abs(Close_i - VWAP_i) / (ATR_i + eps)
```

4. Reversion ratio:

```text
vwap_reversion_ratio(d) =
  count(norm_i <= VWAP_REVERSION_ATR_K) / count(valid_i)
```

where `VWAP_REVERSION_ATR_K = 0.5`.

Feature:

```text
vwap_reversion_gap =
  vwap_reversion_ratio(D-1) - mean_prev(vwap_reversion_ratio)
```

### 6.24 `range_box_stability_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. Collect all rolling 30-minute ranges:

```text
range_w = max(High_window) - min(Low_window)
```

2. Compute:

```text
range_box_stability(d) =
  1 - IQR(range_w) / (median(range_w) + eps)
```

Feature:

```text
range_box_stability_gap =
  range_box_stability(D-1) - mean_prev(range_box_stability)
```

Higher value means the rolling box widths are more stable.

### 6.25 `failed_breakout_gap` (Ridge: yes)

Source: SPX 1m RTH.

Session metric:

1. For every bar `i` after the first `FAILED_BREAKOUT_WINDOW_MINUTES = 30` bars, build the prior box:

```text
box_high = max(High_{i-30:i-1})
box_low  = min(Low_{i-30:i-1})
```

2. A breakout occurs if:

```text
High_i > box_high  or  Low_i < box_low
```

3. Look forward up to `FAILED_BREAKOUT_RETURN_MINUTES = 10` bars.
4. Breakout is failed if any future close returns inside `[box_low, box_high]`.
5. Session ratio:

```text
failed_breakout_ratio(d) = failed_breakouts / total_breakouts
```

Feature:

```text
failed_breakout_gap =
  failed_breakout_ratio(D-1) - mean_prev(failed_breakout_ratio)
```

### 6.26 `session_range_gap` (Ridge: yes)

Source: SPX 1m RTH + SPX daily ATR.

На сессии `d`:

- `session_range_k(d)` = `max(High) − min(Low)` по k-му RTH-сегменту (90+90+90+120 мин), см. `_session_range_segment_cols_for_session`;
- `ATR14_daily(d)`.

Feature (для строки с target date `D`, используется **D−1**):

```text
session_range_gap =
  session_range_4(D-1) / ATR14(D-1)
  − mean(session_range_1..3(D-1)) / ATR14(D-1)
```

Интерпретация: насколько **поздний** (4-й) сегмент вчерашнего дня был шире/уже относительно среднего раннего диапазона — форма intraday range «к концу дня» vs «утро».

## 7. Missing Values And Leakage Rules

Table build:

- no global interpolation;
- no global fill;
- rows keep feature NaNs if a source is unavailable.

GridSearchCV:

- pipeline uses `SimpleImputer(strategy="mean")`;
- imputer is inside each CV fold, so it is fitted only on fold-train data.

Final Ridge train:

- `inf` -> `NaN`;
- column means are computed on train/full train frame;
- `NaN` filled by train column means;
- remaining missing values filled by `0.0`.

Online inference:

- missing feature values are replaced by saved `x_scaler.mean[i]`;
- this mirrors train-time column-mean fallback.

There is one older helper `_fill_missing_neighbor_mean()` that linearly interpolates and then bfill/ffill. It is used in Stage-1 leave-one-out candidate fitting. The main GridSearch and final train paths do not use global interpolation.

## 8. Ridge Model

Ridge pipeline:

1. Выбрать колонки признаков (или `selected_features`; сегментные `chop_dur_*_*` уже отфильтрованы `CHOP_FEATURES_EXCLUDED_FROM_MODEL`). Таргет по умолчанию — **`chop_dur`** сессии (после `finalize_*` на таблице). Стадированный инференс вызывает `train_chop_model` дополнительно с `target_col=chop_dur_30_k` / `chop_dur_20_k`.
2. Fill missing values as described above.
3. Fit `StandardScaler` on X.
4. Fit separate `StandardScaler` on `chop_dur`.
5. Train `Ridge(alpha)`.
6. Raw prediction is inverse-transformed back to original `chop_dur` scale.

Saved artifacts:

- `chop_regression_model.json`;
- `chop_regression_model.joblib`;
- `chop_regression_hyperparams.json`;
- `chop_regression_table.csv.gz`.

JSON model stores:

- `feature_order`;
- Ridge `coef` and `intercept`;
- `x_scaler.mean/scale`;
- `y_scaler.mean/scale`;
- `hyperparameters.alpha`;
- `train_mse`, `train_mae`, `n_train`;
- `linear_calibration` if calibration could be fitted.

## 9. Ridge Calibration

All Ridge predictions go through `_apply_chop_calibration_and_tail()`.

Current mode:

```text
CHOP_PRIMARY_CALIBRATION_MODE = "scale"
```

### 9.1 Raw Mode

`mode="raw"`:

```text
pred = clip(y_pred_raw, 0, 1)
```

### 9.2 Scale Mode (Current Primary)

`mode="scale"` applies mean-preserving variance calibration only:

```text
pred = mean_y_train + variance_k * (y_pred_raw - mean_y_pred_raw_train)
pred = clip(pred, 0, 1)
```

It does not use linear `a,b`.

`variance_k` is built in `_extend_chop_calibration_variance()`:

- raw std ratio = `std(y_true) / std(y_pred_raw_train)`;
- effective variance stretch = `1 + 0.3 * (raw_k - 1)`;
- stores:
  - `mean_y_train`
  - `mean_y_pred_raw_train`
  - `mean_y_lin_train`
  - `variance_k`
  - `variance_k_raw`

Scale mode is used in:

- offline selection objective;
- финальное обучение сохранённой сессионной Ridge (артефакт JSON) и путь `predict_chop_ratio()`;
- **каждая** временная Ridge внутри `_chop_staged_predict_session_chop` (через `train_chop_model` / `predict_chop_ratio_from_payload` с тем же `_apply_chop_calibration_and_tail`).

### 9.3 Full Mode

`mode="full"` exists but is not current primary.

Sequence:

1. Linear calibration:
   `y_lin = a * y_pred_raw + b`.
2. Variance step:
   `mean_y_train + variance_k * (y_lin - mean_y_lin_train)`.
3. Shrink toward train median:
   `median + 0.65 * (pred - median)`.
4. Optional train quantile clip q02/q98 if `CHOP_APPLY_TRAIN_CHOP_DUR_QUANTILE_CLIP=True`.
5. Optional tail adjustments.
6. Final clip `[0,1]`.

Tail adjustment is effectively disabled in current constants:

- `CHOP_PRED_HIGH_TAIL_K = 0.0`;
- `CHOP_PRED_LOW_TAIL_K = 0.0`.

### 9.4 Linear Calibration Fit

`_compute_chop_linear_calibration()`:

- uses only the latest `CHOP_LINEAR_CALIB_MAX_SESSIONS` sessions;
- winsorizes actual `chop_dur` to q10/q90 for robust fit;
- uses recency weights:
  - newest `CHOP_LINEAR_CALIB_FULL_WEIGHT_SESSIONS` sessions: 1.0;
  - next `CHOP_LINEAR_CALIB_SECOND_TIER_SESSIONS`: 0.9;
  - older sessions decay by `CHOP_LINEAR_CALIB_TAIL_DROP` every `CHOP_LINEAR_CALIB_TAIL_DROP_BLOCK`;
- fits weighted first-degree polynomial: `actual ~= a * pred + b`.

Even in `scale` mode, this calibration payload is saved because variance metadata is extended on the same calibration dict.

## 10. Offline Ridge Feature Selection

Offline mode is entered when:

- `RUN_OFFLINE_SELECTION=True`.

In `RUN_OFFLINE_SELECTION=True`, Stage 1 is always rebuilt and existing `chop_regression_hyperparams.json` is overwritten. Saved hyperparams are reused only in daily mode (`RUN_OFFLINE_SELECTION=False`).

### 10.1 Train/Test Split

`build_chop_table(..., n_test_sessions=N_TEST_SESSIONS)`:

- `train_dates` = все целевые даты, кроме последних `N_TEST_SESSIONS` (**27** в текущем коде);
- `test_dates` = последние **`N_TEST_SESSIONS`** целевых дат.

### 10.2 Alpha Search

`grid_search_chop_hyperparameters()`:

- feature set = current candidate feature set;
- model = `SimpleImputer(mean) -> StandardScaler -> Ridge`;
- target `chop_dur` is scaled by separate `StandardScaler`;
- CV = `TimeSeriesSplit(n_splits=5)`;
- scoring = `neg_mean_absolute_error`;
- alpha grid = **`RIDGE_ALPHA_GRID`**, т.е. `[50.0, 60.0, 70.0, 80.0, 90.0, 100.0]`;
- if date column exists, only first **`N_SESSIONS_FOR_HP`** (80) sessions are used for HP search.

### 10.3 Stage-1 Iteration

Число **прогонов** Stage 1 не фиксировано константой: оно равно **`_chop_feature_selection_total_runs(n_start, MIN_FEATURES_AFTER_SELECTION)`**, где `n_start` — число модельных признаков на первом шаге (обычно все доступные из `CHOP_FEATURE_ORDER` минус исключения). Это **число leave-one-out удалений** (`n_start - MIN`) плюс **один** финальный прогон, когда дальнейший дроп не выполняется.

At each run:

1. Choose alpha for the current feature set.
2. Train Ridge on `reg_df_train`.
3. Build calibration on train raw predictions.
4. Apply current primary calibration to train and test predictions.
5. Compute metrics:
   - train MAE/MSE;
   - test MAE/MSE;
   - test direction accuracy;
   - test Spearman;
   - test mono bucket score;
   - **test bucket penalty** (сумма положительных «ступеней» вниз между средними фактами по корзинам).
6. If feature count > `MIN_FEATURES_AFTER_SELECTION`, try removing each feature once.
7. Remove the feature whose absence gives the best calibrated holdout key.

Leave-one-out removal key (минимизация кортежа; приоритет ухудшения при чтении слева направо):

```text
max MonoBuckets -> max Spearman -> max DirAcc -> min BucketPenalty -> min MAE
```

The feature set is then updated and the next run starts.

### 10.4 Stage-1 Metrics

`_chop_test_selection_metrics()`:

`DirAcc`:

- threshold = train median `chop_dur`;
- compares `sign(pred - threshold)` with `sign(actual - threshold)`.

`Spearman`:

- Spearman correlation between calibrated prediction and actual `chop_dur`.

`MonoBuckets`:

- sort holdout rows by prediction;
- split into 6 buckets;
- compute mean actual in each bucket;
- score = `1 - violations / (n_buckets - 1)`;
- a violation is `mean_actual[b] < mean_actual[b-1]`;
- max score is `1.0`.

`BucketPenalty` (lower better):

- те же 6 корзин и средние \(\overline{y}_b\) по фактическому `chop_dur`;
- `bucket_penalty = sum(max(0, mean[b-1] - mean[b]) for b = 1..5)`;
- если данных для корзин недостаточно — `nan`.

Final run selection key:

```text
max MonoBuckets -> max Spearman -> max DirAcc -> min BucketPenalty -> min MAE
```

The chosen config is saved to `chop_regression_hyperparams.json`.

## 11. Historical Validation And Diagnostics

`build_chop_historical_validation_table()`:

- applies the saved Ridge model to historical rows;
- returns actual and predicted chop values for validation;
- receives the already-built regression table and does not rebuild features from raw CSV.

`evaluate_chop_model_quality()`:

- computes MAE/MSE;
- computes direction accuracy around train median;
- computes Spearman if scipy is available;
- prints bucket analysis only when `include_buckets=True`;
- supports scatter plot via `save_chop_pred_actual_scatter_png()`.

`compute_ridge_chop_cv_stability()`:

- runs TimeSeriesSplit stability on chosen feature set/alpha;
- reports mean/std for MAE, DirAcc, Spearman, MonoBuckets.

## 12. CatBoost Experiment And Secondary Model

CatBoost is optional and only active when `RUN_OFFLINE_SELECTION=True` and `CHOP_OFFLINE_INCLUDE_CATBOOST=True`.

`run_catboost_chop_experiment()`:

- does not replace production Ridge artifacts;
- uses eligible chop features (те же исключения `CHOP_FEATURES_EXCLUDED_FROM_MODEL`, что у Ridge);
- performs CatBoost parameter grid search with `TimeSeriesSplit`;
- runs leave-one-out pruning с той же логикой **динамического** числа прогонов, что Ridge Stage 1 (`_chop_feature_selection_total_runs`);
- после выбора лучшего набора печатает **rolling** метрики на скользящих окнах **`train_n→test_n`**: **`(60, 15), (75, 20), (85, 25)`**;
- calibrates predictions with CatBoost-specific calibration (§12.1);
- saves `chop_catboost_experiment.json`.

### 12.1 CatBoost calibration modes

Поддерживаются три режима (`CHOP_CATBOOST_CALIBRATION_MODE`): **`raw`**, **`scale`**, **`quantile`** (дефолт в коде — **`quantile`**).

**`scale`** (fallback):

```text
pred = mean_y_train + variance_k * (raw - mean_y_pred_raw_train)
pred = clip(pred, 0, 1)
```

`variance_k` ограничивается `[CHOP_CATBOOST_VAR_K_MIN, CHOP_CATBOOST_VAR_K_MAX]`.

**`quantile`** — **не** pointwise-маппинг по отсортированным парам `(raw_pred_train_i, y_train_i)` (это давало бы train MAE ≈ 0). Вместо этого:

1. На обучающей выборке строятся **квантильные бины** по сырым предсказаниям модели (**равное число точек в бине**, целевое число бинов `CHOP_CATBOOST_QUANTILE_N_BINS`, по умолчанию **7**, с нижней границей **3** и ограничением по объёму выборки).
2. В каждом бине: **`x_bin = median(raw_pred)`**, **`y_bin = median(actual)`**.
3. По точкам `(x_bin, y_bin)` в порядке возрастания `x` выполняется **монотонное сглаживание** `y` через **`IsotonicRegression`** (если sklearn недоступен — простой монотонный проход по `y`).
4. Для новых предсказаний используется **кусочно-линейная** (и при возможности PCHIP) интерполяция по сохранённым узлам — поля артефакта `quantile_x_pred_sorted` / `quantile_y_actual_sorted`.
5. Если **`n_calib < CHOP_CATBOOST_QUANTILE_MIN_N` (40)** — квантильная калибровка **не** применяется: для режима `quantile` выполняется fallback на **`scale`**, при невозможности scale — на **`raw`**.
6. **Median shrink:** `final = median_train + shrink_k * (mapped - median_train)`; сетка **`CHOP_CATBOOST_SHRINK_GRID = (0.70, 0.80, 0.90, 1.00)`**.
7. **`shrink_k` подбирается только по out-of-fold предсказаниям** (`_catboost_oof_raw_predictions`: тот же CatBoost и параметры, **`TimeSeriesSplit`** на матрице признаков train). На OOF считается MAE для каждого `shrink_k`. Если валидных OOF-пар меньше **`CHOP_CATBOOST_SHRINK_OOF_MIN_N` (20)** или OOF не передан — используется **`CHOP_CATBOOST_SHRINK_FALLBACK_K = 0.85`**.

Режим **`tail`** (legacy): после шага `scale` возможна асимметричная «растяжка» хвостов относительно медианы обучения (используется только если явно задан в конфиге/артефакте).

`train_catboost_chop_model()`:

- обучает финальную CatBoost на всех строках;
- перед сохранением калибровки считает **OOF** на полной матрице для выбора **`shrink_k`**;
- сохраняет `chop_catboost_model.json` и `.joblib` (feature means, params, `linear_calibration`, `calibration_mode`, importances).

`predict_catboost_chop_ratio()`:

- загружает joblib CatBoost;
- заполняет пропуски сохранёнными средними по признакам;
- применяет сохранённую калибровку (`_apply_catboost_chop_calibration`).

В текущем дефолте репозитория **`CHOP_OFFLINE_INCLUDE_CATBOOST=False`** — ветка CatBoost не запускается, пока константа не переведена в **`True`** (при **`RUN_OFFLINE_SELECTION=True`**).

## 13. Daily Retrain And Inference

`RUN_OFFLINE_SELECTION=False` is the daily production mode.

Daily requirements:

- valid `chop_regression_hyperparams.json`;
- source data files available;
- cached `chop_regression_table.csv.gz` if available. If the cache is missing, the module builds it once.

Daily sequence in `__main__`:

1. Load SPX/ES/VIX/VIX9D data.
2. Load/update cached regression table:
   - load `chop_regression_table.csv.gz`;
   - compare cached `date` values with available SPX target sessions;
   - build only missing target rows through `_build_chop_rows_for_dates()`;
   - append missing rows and save the cache again.
3. Load saved `selected_features` and `alpha` from `chop_regression_hyperparams.json`.
4. Retrain final Ridge on all rows of the updated table with unchanged structure:
   `train_chop_model(reg_df_daily, alpha=saved_alpha, feature_cols=saved_features)`.
5. Determine the next target date after the last SPX RTH session.
6. Try to get `open_pred` from `open_price.predict_spx_open()`.
7. Call `predict_session_chop(..., skip_holdout_training=True, fast_inference=True)`.
8. Print the prediction and save the usual `chop_prediction_<date>.json`.
9. Build validation from the already-loaded table:
   `build_chop_historical_validation_table(reg_df_daily)`.
10. Print short quality summary through `evaluate_chop_model_quality(..., include_buckets=False)`.

Daily mode does not:

- run feature selection;
- change `selected_features`;
- change `alpha`;
- rebuild the full regression table if cache exists;
- rebuild historical feature rows before validation;
- print bucket analysis.

`predict_session_chop()` (сжатый порядок, см. код):

1. Загрузка данных `_load_chop_data()`.
2. `update_chop_regression_table_cache_incremental(...)` (опционально исключить `exclude_session_from_regression_table` — та же дата, что `--plan-for` в `main`).
3. При невалидных гиперпараметрах и `fast_inference=False` может вызываться `grid_search_chop_hyperparameters` (в обычном быстром режиме — нет).
4. Срез дат D-1…D-5 по SPX; при необходимости `predict_bias_and_probability` → `bias_dir_4`/`bias_strength_4` (или override).
5. `build_chop_features(..., exclude_spx_rth_on_date=target_date)` — без RTH целевого дня на SPX.
6. **`_chop_staged_predict_session_chop`** → `predicted_chop_pct` и по-сегментные проценты (см. [§5.1](#51-staged-chop-at-inference)); при intraday-аргументах — фиксированные стадии.
7. Для EM/targets/similarity: из словаря признаков берутся **`pred_pm_range_norm`** (`pm_range_norm`), **`pred_pm_iv_rv_dislocation`** (`pm_iv_rv_dislocation`), **`pred_pm_vwap_stickiness`** (`pm_vwap_stickiness`) и числовой **`pred_bias`**; **pred bias** — сначала `_load_bias_for_chop_from_debug(target_date)` (`bias_last_prediction_debug.json`, ключи `bias_for_chop` / `bias_strength_for_chop`), иначе из словаря bias `bias_for_chop` / `bias_strength_for_chop`. Также передаются **`forecast_session_date`** и упорядоченный список торговых дат для возраста сессий во временном весе.
8. `get_em_from_chop_calibration` и `compute_target_of_session_from_similar_chop` (при конечном `open_pred`).
9. Построение зон (weighted / fallback `detect_chop_zones_v2`).
10. `save_chop_result`; опционально CatBoost comparison через `predict_catboost_chop_ratio` + `_compute_chop_metrics_for_prediction`.

11. Если из `compute_target_of_session_from_similar_chop` получены словари весов `w_bull` / `w_bear`, по ним дополнительно считается **`similarity_sequence_range`** и кладётся в JSON как **`sequence+range`** ([§17](#17-sequencerange-co-sign-runs-and-hl-range-on-similar-sessions)).

**Не путать:** основной `predicted_chop_pct` идёт из стадированного Ridge, а не из вызова `predict_chop_ratio()` по одному JSON.

`predict_chop_ratio()` вручную применяет сохранённый Ridge JSON (валидация, сравнения, утилиты):

```text
x = features in model.feature_order
missing x -> x_scaler.mean[i]
x_scaled = (x - mean) / scale
y_scaled = dot(x_scaled, coef) + intercept
y_raw = y_scaled * y_scale + y_mean
pred = _apply_chop_calibration_and_tail(y_raw, linear_calibration)
```

## 14. Similarity Pool For EM And Targets

EM и targets используют `_weighted_similarity_pool_for_targets()`, когда переданы **все** текущие компоненты контекста (см. `get_em_from_chop_calibration`: иначе пул только по **жёсткому safety-cap по chop%** без экспоненциальной similarity).

Context values (прогноз «текущей» сессии; часть полей не идёт в Ridge, но нужна для пула):

- predicted chop percent;
- **`pred_bias`** — один скаляр для расстояния по bias ([§14.1](#141-bias-combined-vs-pred_bias-v-similarity));
- **`pred_pm_range_norm`** ← `pm_range_norm` из текущего словаря признаков;
- **`pred_pm_iv_rv_dislocation`** ← `pm_iv_rv_dislocation`;
- **`pred_pm_vwap_stickiness`** ← `pm_vwap_stickiness`;
- **`forecast_session_date`** и **`trading_dates_ordered`** — для `age_sessions` и временного затухания.

### 14.1 `bias_combined` vs `pred_bias` в similarity

Это разные роли; **сырые `bias_dir` и `bias_strength` в расстоянии по bias не смешиваются по двум осям** — везде участвует **одна числовая шкала на строку**.

**Текущая сессия:** `_load_bias_for_chop_from_debug` читает `bias_last_prediction_debug.json` (при совпадении `predict_date` с целевой датой). Сначала **`bias_for_chop`**, иначе **`bias_strength_for_chop`** — это **`pred_bias`** в `_weighted_similarity_pool_for_targets`.

**Исторические строки:** **`_pool_bias_strength_series`** — предпочтительно **`bias_combined`**, иначе `bias_strength` → `y_bias` → `bias`. Расстояние по bias: \(|hist\_scalar - pred\_bias|\).

**Bull/bear-веса строк пула:** знак исторического дня — у той же серии (обычно **`bias_combined`**), не у текущего `pred_bias` ([§15](#15-em-calculation)).

См. шаг 7 в [§13](#13-daily-retrain-and-inference).

Жёсткий отбор по chop:

- маска **`abs(chop_i% - predicted_chop_pct) <= chop_safety_cap_pp`** с базовым **`CHOP_SIM_CHOP_SAFETY_CAP_PP = 25`** п.п. (ещё один порядок отбора — вес **`z_chop`** по робастной шкале, см. ниже). Константы **`CHOP_SIM_BAND_*`** (10→12→15) **удалены**.
- если передан кортеж **`predicted_chop_seg_pct`** (четыре процента по сегментам) и в таблице есть **`chop_dur_1`…`chop_dur_4`**, расстояние по chop строится как **минутно-взвешенное** среднее \(|hist_k\% - pred_k\%|\) (`_weighted_segment_chop_distance_pp`, веса 90/90/90/120); иначе — по сессионному `chop_dur` × 100, как раньше.

Similarity distance (**робастные z по пулу** после маски):

- **`z_chop`** по \(|chop\% - pred|\);
- **`z_bias`** по \(|hist\_scalar - pred\_bias|\), где `hist_scalar` из **`bias_combined`** строки (fallback см. [§14.1](#141-bias-combined-vs-pred_bias-v-similarity));
- **`z_pm_iv_rv_dislocation`** по \(|pm\_iv\_rv\_dislocation^{hist} - pred\_pm\_iv\_rv\_dislocation|\);
- **`z_pm_range_norm`** по \(|pm\_range\_norm^{hist} - pred\_pm\_range\_norm|\);
- **`z_pm_vwap_stickiness`** по \(|pm\_vwap\_stickiness^{hist} - pred\_pm\_vwap\_stickiness|\).

Линейная форма расстояния (коэффициенты — **`CHOP_SIM_W_*`**):

```text
d_new =
  CHOP_SIM_W_CHOP * z_chop +
  CHOP_SIM_W_BIAS * z_bias +
  CHOP_SIM_W_PM_IV_RV_DISLOCATION * z_pm_iv_rv +
  CHOP_SIM_W_PM_RANGE_NORM * z_pm_range_norm +
  CHOP_SIM_W_PM_VWAP * z_pm_vwap

w_similarity = exp(-similarity_coef_scale * d_new)
```

Текущие веса (утро / base pool, без intraday anchors):

- **`CHOP_SIM_W_CHOP = 0.45`**
- **`CHOP_SIM_W_BIAS = 0.20`**
- **`CHOP_SIM_W_PM_IV_RV_DISLOCATION = 0.10`**
- **`CHOP_SIM_W_PM_RANGE_NORM = 0.15`**
- **`CHOP_SIM_W_PM_VWAP = 0.10`**

Временное затухание и итоговый вес строки:

```text
age_i = число сессий от текущего якоря (последняя SPX-дата строго до forecast_session_date) до даты строки i
w_time_i = exp(-age_i / tau)          с базовым tau = CHOP_SIM_TEMPORAL_TAU_SESSIONS (= 24)
w_raw_i = w_similarity_i * w_time_i
w_cap = quantile(w_raw, CHOP_SIM_WEIGHT_CAP_QUANTILE)   (= 0.95)
w_total_i = min(w_raw_i, w_cap)
```

Bull/bear для EM/targets: **`w_bull_i = w_total_i * max(0, bias_i)`**, **`w_bear_i = w_total_i * max(0, -bias_i)`**, где **`bias_i`** — тот же signed скаляр, что **`_pool_bias_strength_series`** (обычно `bias_combined`).

**`_compute_group_weighted_stats`**: если переданы **`similarity_weight_by_session_date`**, временный **`decay_lambda` для recency принудительно 0**, чтобы **не удваивать** temporal decay (он уже в **`w_time`**).

ESS и эскалация при **`ESS < CHOP_ESS_TARGET_MIN`** (**`7.0`**):

1. базовый проход: `similarity_coef_scale=1`, `tau` базовый, cap **25** п.п.;
2. **`similarity_coef_scale *= 0.75`**;
3. то же + **`tau *= 1.5`**;
4. то же + **`chop_safety_cap_pp = CHOP_SIM_CHOP_SAFETY_CAP_EXPAND_PP`** (**30** п.п.).

### 14.2 Intraday `session_range` anchors and post-filter

Только при вызове из **`update_TradePlan`** / `reapply_intraday_chop_session_result` с непустым **`intraday_pool_anchors`**:

```text
intraday_pool_anchors = {
  "n_segments": k,                          # 1..3 завершённых RTH-сегмента
  "session_range_pts": {1: r1, 2: r2, …},   # SPX pts, High−Low по сегменту
  "chop_actuals_frac": {…}                  # опционально, для audit
}
```

**Утренний** прогон (`predict_session_chop` без anchors): поведение **без изменений** — нет веса по `session_range`, нет post-filter.

**Intraday update** (`_resolve_intraday_similarity_pool_for_targets`):

1. **Базовый similarity-пул** строится с `intraday_range_soft_weight=True`, т.е. range-anchor **всегда** участвует в `d_new` до post-filter.  
   На intraday пути используются отдельные веса **`CHOP_SIM_UPDATE_W_*`** (см. ниже), а range-anchor добавляется как  
   **`+ CHOP_SIM_W_INTRADAY_RANGE_SEG * z_range_anchor`**.
2. **Post-filter (предпочтительный путь):** оставить строки, где для каждого `k ≤ n_segments` выполняется  
   `live_k / band ≤ hist session_range_k ≤ live_k * band` с **`band = 1.25`**, при `min(ESS_bull, ESS_bear) ≤ 5` — повтор с **`band = 1.40`**.
3. Если после фильтра ESS всё ещё низкий — **fallback** на pool с `intraday_range_soft_weight=True` **без** hard filter (soft-weight pool).

**Веса intraday update (когда `intraday_range_soft_weight=True`):**

- `CHOP_SIM_W_INTRADAY_RANGE_SEG = 0.30`
- `CHOP_SIM_UPDATE_W_CHOP = 0.50`
- `CHOP_SIM_UPDATE_W_BIAS = 0.10`
- `CHOP_SIM_UPDATE_W_PM_IV_RV_DISLOCATION = 0.00`
- `CHOP_SIM_UPDATE_W_PM_RANGE_NORM = 0.10`
- `CHOP_SIM_UPDATE_W_PM_VWAP = 0.00`

Формула ESS по ненормализованным весам диапазона:

```text
ESS = (sum(w)^2) / sum(w^2)
```

## 15. EM Calculation

`get_em_from_chop_calibration()` returns:

- `mean_range_spx_bull`;
- `mean_range_spx_bear`;
- diagnostics dict.

Algorithm:

1. Если задан полный контекст (все предикторы similarity конечны) — построить weighted similarity pool (как в [§14](#14-similarity-pool-for-em-and-targets)); иначе маска только по **`predicted_chop_pct`** с тем же **safety cap** по пунктам процентов (`__w_sim__` = 1 на прошедших маску строках).
2. Взять исторический диапазон строк пула: `actual_high_spx - actual_low_spx`.
3. Winsorize по **q05/q95** на значениях диапазона пула (`range_val_w`).
4. **Разделение bull/bear по историческому bias строки**, а не по текущему `pred_bias`:  
   `bh = _pool_bias_strength_series(pool)` (предпочтительно **`bias_combined`**, затем `bias_strength`, затем наследие `y_bias` / `bias`):  
   `w_bull_row = w_sim * max(0, bh)`, `w_bear_row = w_sim * max(0, -bh)`.
5. Взвешенное среднее по датам: если переданы **similarity-веса по датам**, используется **`decay_lambda = 0`** (recency уже в **`w_time`** пула); иначе — прежний адаптивный **recency decay** `lambda` (`0.10` / `0.07` / `0.04` от размера группы) → **`EM_raw`** (bull/bear).
6. Если на стороне нет массы — fallback на `_em_one(work["w_sim"])` (без bull/bear множителя).
7. **Якорение EM к референсному диапазону R (per side):**  
   - `pool_median_side` — медиана winsorized `range_val_w` по строкам с положительным `w_bull` / `w_bear`;  
   - `recent_median_side` — медиана `actual_range_spx` за последние **15** торговых дат (bull: bias>0, bear: bias<0);  
   - `R = CHOP_EM_ANCHOR_W_POOL * pool_median + CHOP_EM_ANCHOR_W_RECENT * recent_median` (**0.30 / 0.70**);  
   - `EM_final = EM_raw ± CHOP_EM_ANCHOR_BETA * |EM_raw − R|` (к **R**, beta **0.60**; `_anchor_em_raw_to_reference`).
8. Проверка ESS по bull/bear; при низкой — те же **четыре уровня эскалации**, что для targets ([§14](#14-similarity-pool-for-em-and-targets)): baseline → **`similarity_coef_scale×0.75`** → **`tau×1.5`** → **cap 25→30 п.п.** (`ess_fallback_tier_em` и поля масштабов в diagnostics).

**Устарело / удалено:** прежний **regime scaling** `recent_range / pool_range` в EM больше **не** применяется.

Агрегированное **для сохранённого результата** `mean_range_spx` внутри `predict_session_chop` — **полусумма bull и bear**, если оба есть; иначе доступная сторона (не `max`; см. код `predict_session_chop` после вызова `get_em_from_chop_calibration`).

Примечание: `pred_bias` используется **только для построения similarity-пула**, не как множитель `w_bull`/`w_bear` строк EM.

## 16. Scenario Targets

`compute_target_of_session_from_similar_chop()` returns:

- `target_bullish_high`;
- `target_bullish_low`;
- `target_bearish_high`;
- `target_bearish_low`;
- `mean_ratio_bull_high`;
- `mean_ratio_bull_low`;
- `mean_ratio_bear_high`;
- `mean_ratio_bear_low`.

For each historical session in similarity pool:

```text
range_i = High_i - Low_i
ratio_high_i = (High_i - Open_i) / range_i
ratio_low_i  = (Open_i - Low_i) / range_i
```

Side weights (по каждому сеансу `i` в похожем пулу) используют **исторический** signed bias строки таргета, тот же приоритет колонок, что `_pool_bias_strength_series`:

```text
bias_i из bias_combined (или fallback bias_strength / …)
w_bull_i = w_total_i * max(0, bias_i)
w_bear_i = w_total_i * max(0, -bias_i)
```

Текущий `pred_bias` участвует в **расстоянии** similarity (сопоставление с историческими bias), см. `_weighted_similarity_pool_for_targets`.

Для каждой стороны (bull / bear):

- compute weighted historical range;
- compute weighted `ratio_high`;
- compute weighted `ratio_low`;
- combine EM side range and target group range:

```text
range_scale = 0.3 * mean_range_em_raw + 0.7 * weighted_range_group
range_scale = max(range_scale, mean_range_em_raw)   # when em_raw is set
```

- **Anchor `targets_range_scale` (per side)** toward the same reference blend **R** as EM ([§15](#15-em-calculation)), with **`CHOP_TARGETS_ANCHOR_BETA = 0.20`** (`_anchor_em_raw_to_reference` on `targets_range_scale_bull/bear` after the max-with-em_raw step).

Targets:

```text
target_high = open_pred + range_scale * weighted_ratio_high
target_low  = open_pred - range_scale * weighted_ratio_low
```

**Expected Range vs Max Target envelope** (`enforce_chop_targets_cover_expected_range`, утренний и intraday reapply):

- Expected Range bounds = chop EM (winsor × ratios) via `expected_range_lo_hi_from_chop_em`;
- scenario **`target_*`** (Max Target pipeline) **расширяются**, если уже посчитанные targets уже не покрывают ER (только расширение, ER не меняется).

Fallbacks:

- if side-specific weights fail, use `w_total`;
- if EM side range is missing, use weighted group range;
- if weighted group range is missing, use EM side range.

The function also exports `zone_weights_out`:

- `w_bull`;
- `w_bear`;
- `w_total`;
- dates used.

These weights are reused by weighted chop-zone detection and by **`sequence+range`** ([§17](#17-sequencerange-co-sign-runs-and-hl-range-on-similar-sessions)).

## 17. `sequence+range`: CO sign runs and HL range on similar sessions

Блок **не входит в Ridge** и не использует `chop_dur`. Это отдельная метрика «микроструктуры» внутридневного ряда SPX: по **тем же датам и весам**, что и scenario targets (`w_bull`, `w_bear` из `compute_target_of_session_from_similar_chop`), на каждой исторической RTH-сессии строятся **пробеги (runs) согласованного знака тела минутной свечи**, затем отбираются типичные по длине сегменты и по ним считаются **взвешенные по similarity квантили q35/q65** длины (в барах) и суммарного High−Low пробега.

### 17.1 Где считается и куда сохраняется

- Реализация: `compute_similarity_sequence_range_payload(spx_1m, w_bull, w_bear)` и вспомогательные функции с префиксом `_chop_session_co_run_records` / `_pool_sequence_aggregate`.
- Вызывается в конце `predict_session_chop()` (после targets/zones), если `zone_weights_for_targets` содержит словари `w_bull` и `w_bear` и есть SPX 1m.
- Поле результата: `ChopSessionResult.similarity_sequence_range`.
- В `chop_prediction_<date>.json` дублируется под ключом **`sequence+range`** (и внутри `model_metrics` для Ridge CatBoost-сравнения при наличии).

### 17.2 Знак тела свечи и границы пробегов

Источник: **SPX 1m RTH**, строка торгового дня `d`.

1. **Первая RTH-свеча сессии не участвует** в сегментации (как и в коде комментария к `_chop_session_co_run_records`: отсчёт со второго RTH-бара).
2. Для каждого бара: эффективный знак `sign(Close − Open)`; нули маскируются и замещаются **ffill → bfill → +1**.
3. **Пробеги с допуском на шум**: подряд идущие бары считаются одним пробегом, пока не накопится **третья подряд** свеча с противоположным знаком (**до двух** «чужих» знаков внутри пробега поглощаются). Это `_chop_segment_sign_run_bounds`.
4. На каждый пробег: длина в барах `len_bars`, суммарный диапазон сегмента `range_hl = max(High) − min(Low)` по барам пробега, метка знака **`+`** или **`−`** по знаку первого бара пробега.

Интерпретация для Trade Plan и отчётов:

- **plus (+)** ↔ последовательность баров доминирующего «ап» по телу (**sequence_UP / range_UP** в пользовательских ячейках).
- **minus (−)** ↔ **sequence_DOWN / range_DOWN**.

### 17.3 Внутрисессионный фильтр по типичной длине пробегов

Не все пробеги дня попадают в пул — на каждой сессии `d` отдельно для знака **`+`** и **`−`**:

1. Собираются все `(len_bars, range_hl)` с нужным знаком.
2. По множеству длин считаются квантили сессии:
   - по умолчанию **\[q40, q60\]** длины: `CHOP_RUN_SEQ_SEGMENTS_PRIMARY_Q`;
   - если таких пробегов **меньше** `CHOP_RUN_SEQ_FALLBACK_THRESHOLD_N` (15), используется полоса **\[q30, q70\]**: `CHOP_RUN_SEQ_SEGMENTS_FALLBACK_Q`.
3. Остаются только пробеги, у которых **длина попадает в эту полосу** (`_filter_runs_by_within_session_quantile`).  
   То есть на каждый исторический день берутся преимущественно «середнячки» по длительности аналогично направленных пробегов, а их `range_hl` идёт вместе с длиной.

### 17.4 Агрегация по similarity-пулу (bull и bear отдельно)

Для **стороны similarity** (`w_bull` или `w_bear`):

1. Для каждой даты `d` с положительным весом `w(d)` (после нормализации ключа через `_chop_normalize_session_date_weight_key`) строятся отфильтрованные списки длин и HL-диапазонов для `+` и для `−`.
2. Каждый сохранённый пробег умножается на **`w(d)` одной и той же сессии** — все пробеги дня имеют один вес сеанса (`_pool_sequence_aggregate`).
3. По объединённым спискам (по знаку **`+`** и по **`−`** в отдельности) считаются **взвешенные квантили** с вероятностями **`CHOP_RUN_SEQ_CROSS_SESSION_QLOW`** = **0.35** и **`CHOP_RUN_SEQ_CROSS_SESSION_QHIGH`** = **0.65** (`_weighted_quantile_interp` — линейная интерполяция по упорядоченным значениям и нормализованной CDF весов).
4. Результат по каждой стороне и знаку — два словаря **`q35`**, **`q65`** (или `null`, если данных нет):
   - для **bull-пула**: `bull_plus_run_len_summary`, `bull_plus_run_range_summary`, `bull_minus_run_len_summary`, `bull_minus_run_range_summary`;
   - для **bear-пула**: те же ключи с префиксом `bear_`.

Структура JSON:

```text
sequence+range: {
  "bull": { bull_* summaries },
  "bear": { bear_* summaries }
}
```

Константы порога веса: `CHOP_RUN_SEQ_WEIGHT_EPS = 1e-18` (отсекаются нулевые/нечисловые веса).

### 17.5 Связь с EM, targets и Trade Plan

- **Те же `w_bull` / `w_bear`**, что и для weighted chop-zones и targets; текущий `pred_bias` в этой метрике **не** участвует напрямую — только через то, как построен similarity-пул и side-веса строк.
- При отсутствии валидных словарей весов или ошибке расчёта поле остаётся **`null`** / `similarity_sequence_range = None`.
- Downstream (`build_TradePlan` и интрадей-обновление) может отображать интервалы q35–q65 как ориентиры **типичной длины последовательности баров одного знака** и **типичного суммарного HL-диапазона** таких пробегов в контексте похожих сессий.

## 18. Chop Zones

Chop zones are price intervals derived from historical flat segments.

### 18.1 Historical Flat Segments

`_flat_segments_spx_rth_day()`:

1. Uses SPX RTH 1m bars.
2. Computes ATR(14).
3. Finds raw 30m windows passing K1/K2/K3 flat filters.
4. Merges raw windows if gap <= `MERGE_GAP_MINUTES = 3`.
5. Drops segments shorter than `MIN_ZONE_DURATION_MIN = 30`.
6. For each segment:
   - `z_lo = q10(Close_segment)`;
   - `z_hi = q90(Close_segment)`;
   - `inside_share = share(Close in [z_lo, z_hi])`.
7. Keeps segment only if `inside_share >= INSIDE_SHARE_MIN = 0.8`.
8. Stores date, start/end, zone bounds, center, width, duration, inside share, ATR reference.

### 18.2 Unweighted Zones

`detect_chop_zones_v2()`:

Search band priority:

1. If caller passes finite `pred_search_lo/pred_search_hi`, use that band.
2. Else fallback to:

```text
open_pred +/- CHOP_RANGE_MULT * mean_range_spx
```

with `CHOP_RANGE_MULT = 1.25`.

Historical segment passes if:

```text
overlap(segment, search_band) / segment_width >= ZONE_PRED_OVERLAP_PCT
```

where `ZONE_PRED_OVERLAP_PCT = 0.50`.

Merging:

- zones merge if center distance <= `max(MERGE_CENTER_TOL_MIN, MERGE_CENTER_TOL_MULT * median_width)`;
- and overlap ratio >= `MERGE_OVERLAP_MIN`.

Ranking:

- sort by `session_frequency` descending;
- then `segment_frequency` descending;
- return top `MAX_ZONES_RETURN = 6`.

### 18.3 Weighted Zones

The main online path uses `detect_chop_zones_weighted()` when target similarity weights are available.

Inputs:

- price band;
- session weights by date;
- label: `"bull"`, `"bear"`, `"union"`, or fallback labels.

Algorithm:

1. Collect historical flat zones only for weighted similar dates.
2. For each segment:
   - compute overlap with search band;
   - compute distance from segment center to band center;
   - allow soft inclusion if segment center is near the band even with zero overlap;
   - attach session weight and segment score weight.
3. Merge similar zones.
4. Score merged zone:

```text
score = sum_session_weights * mean_duration * mean_inside_share * score_weight
```

5. Keep only zones with positive score and `score_weight > 0.05`.
6. Rank by:
   - score descending;
   - session frequency descending;
   - segment frequency descending.

Online `predict_session_chop()` builds:

- bullish zones using `w_bull` and bullish target band;
- bearish zones using `w_bear` and bearish target band;
- union zones using combined weights and union target envelope.

If a side returns too few zones, it can fallback to total weights.

### 18.4 Search Bands In Online Flow

If scenario targets exist:

- bullish band = bullish target low/high expanded by `CHOP_ZONE_ENVELOPE_MARGIN_FRAC`;
- bearish band = bearish target low/high expanded by the same margin;
- union band = min(bull_low,bear_low) to max(bull_high,bear_high), also margin-expanded.

`CHOP_ZONE_ENVELOPE_MARGIN_FRAC = 0.002` (0.2%).

If targets do not exist:

- fallback band = `open_pred +/- CHOP_RANGE_MULT * mean_range_spx`.

## 19. Run Modes

Main flow is controlled by:

```text
RUN_OFFLINE_SELECTION = True / False
```

`RUN_OFFLINE_SELECTION=True` (**значение константы в репозитории по умолчанию**; для ежедневного режима без полного offline отбора признаков выставляют **`False`**):

- full research/rebuild mode;
- builds full `reg_df_all` once through `build_chop_table()`;
- saves `chop_regression_table.csv.gz`;
- runs Ridge Stage 1 feature selection and alpha search;
- overwrites `chop_regression_hyperparams.json`;
- trains final Ridge on all rows of the same `reg_df_all`;
- does not rebuild the table again before final train;
- validates final Ridge from the same table through `build_chop_historical_validation_table(reg_df_full)`;
- prints full quality report with buckets;
- predicts one upcoming session and computes EM/targets/zones.

`RUN_OFFLINE_SELECTION=False` (**ежедневный** режим после переключения константы):

- daily retrain mode;
- loads `chop_regression_table.csv.gz`;
- appends only missing target rows;
- loads `selected_features + alpha` from `chop_regression_hyperparams.json`;
- retrains final Ridge on all rows of the updated table;
- does not run feature selection;
- does not rebuild the full regression table if cache exists;
- validates from the updated table;
- prints short quality report without buckets;
- predicts one upcoming session and computes EM/targets/zones.

CatBoost is controlled separately:

```text
CHOP_OFFLINE_INCLUDE_CATBOOST = True / False
```

When `True` and `RUN_OFFLINE_SELECTION=True`, the module additionally runs the CatBoost experiment and final secondary CatBoost artifact. When `False`, the default behavior is Ridge-only.

`CHOP_RUN_MODE` remains in the file as legacy metadata for old configs/logs, but it no longer controls the main branch.

## 20. Saved Result

`save_chop_result()` writes:

- `chop_predictions/chop_prediction_<date>.json`

Main payload:

- `target_date`;
- `predicted_chop_pct`;
- `mean_range_spx`;
- `mean_range_spx_bull`;
- `mean_range_spx_bear`;
- `em_adjusted`;
- `chop_zones`;
- `bullish_chop_zones`;
- `bearish_chop_zones`;
- target high/low for bull and bear;
- mean ratio high/low for bull and bear;
- `features_used`;
- `chop_diagnostics`;
- **`sequence+range`**: вложенная структура `bull` / `bear` с квантильными сводками длин пробегов и HL-диапазонов ([§17](#17-sequencerange-co-sign-runs-and-hl-range-on-similar-sessions)); дублируется в `model_metrics.ridge` (и при сравнении — в ветке catboost);
- optional CatBoost comparison/debug fields when enabled.

`ChopSessionResult` is the in-memory dataclass representation of the same output family; поле **`similarity_sequence_range`** соответствует JSON-ключу **`sequence+range`**.

## 21. Key Functions

Data/calendar:

- `_load_chop_data`
- `_ensure_datetime_tz`
- `_ensure_dt`
- `_get_trading_dates_from_1m`
- `_get_premarket_for_session`
- `_first_target_date_with_es_premarket`

Features/target:

- `build_chop_features`
- `build_chop_table`
- `_build_chop_rows_for_dates`
- `update_chop_regression_table_cache_incremental`
- `calculate_actual_chop_dur`
- `_chop_segment_target_cols_for_session`
- `finalize_chop_segment_and_session_targets_inplace`
- `session_weighted_chop_from_segments`
- `blend_chop30_with_chop20`
- `_chop_add_bias_combined_column`
- `_pool_bias_strength_series`
- `_load_bias_for_chop_from_debug`
- `_chop_dur_flat_window_coverage`
- `_flat_windows_pass_filters`
- gap-признаки / IV (см. **§6.10a**, **§6.7a**, **§6.26**): … `_feature_session_range_gap`, `_session_range_segment_cols_for_session`, …

Ridge:

- `grid_search_chop_hyperparameters`
- `train_chop_model`
- `_chop_base_model_feature_cols`
- `_chop_staged_predict_session_chop`
- `predict_chop_ratio`
- `_compute_chop_linear_calibration`
- `_extend_chop_calibration_variance`
- `_apply_chop_calibration_and_tail`
- `_chop_test_selection_metrics`
- `_pick_best_chop_run_index`
- `_chop_loo_removal_steps`, `_chop_feature_selection_total_runs` (динамическое число прогонов Stage 1)

Validation:

- `build_chop_historical_validation_table`
- `evaluate_chop_model_quality`
- `compute_ridge_chop_cv_stability`

CatBoost:

- `run_catboost_chop_experiment`
- `train_catboost_chop_model`
- `predict_catboost_chop_ratio`
- `_compute_catboost_chop_calibration`
- `_apply_catboost_chop_calibration`
- `_catboost_oof_raw_predictions` (OOF для выбора `shrink_k`)
- `_catboost_binned_isotonic_xy` (бинированная монотонная калибровочная кривая)
- `_catboost_pick_shrink_k_oof`
- `_catboost_interp_actual_given_sorted_preds`

EM/targets/zones:

- `_weighted_similarity_pool_for_targets`
- `_weighted_segment_chop_distance_pp`
- `_resolve_intraday_similarity_pool_for_targets`
- `_filter_pool_by_intraday_session_range`
- `_weighted_intraday_range_anchor_distance`
- `_range_regime_side_reference_pack`, `_anchor_em_raw_to_reference`
- `_chop_session_age_before_forecast`
- `_adaptive_chop_pct_band_mask`
- `_compute_group_weighted_stats`
- `get_em_from_chop_calibration`
- `compute_target_of_session_from_similar_chop`
- `expected_range_lo_hi_from_chop_em`
- `enforce_chop_targets_cover_expected_range`
- `compute_similarity_sequence_range_payload` (…)
- `detect_chop_zones_v2`
- `detect_chop_zones_weighted`
- `_flat_segments_spx_rth_day`
- `_weighted_zone_tuples`

Intraday update ([§22](#22-intraday-chop-update-path)):

- `load_chop_regression_table_cache`
- `chop_intraday_blended_actuals_from_trimmed_session_block`
- `compute_effective_intraday_chop_segments`
- `reapply_intraday_chop_session_result`

Public API:

- `predict_session_chop`
- `save_chop_result`
- `load_chop_result`

## 22. Intraday chop update path

Вызывается из [`update_TradePlan.py`](../update_TradePlan.py) (см. [`update_plan.md`](update_plan.md)). Утренний `predict_session_chop` без intraday-аргументов использует только часть этого пайплайна (сегментный similarity при наличии `predicted_chop_pct_1..4`, без guardrail A2 и без `intraday_pool_anchors`).

### 22.1 Blended segment actuals from live bars

`chop_intraday_blended_actuals_from_trimmed_session_block(rth_block_trimmed, regression_df, n_complete_segments=k)`:

1. RTH-блок уже обрезан по фазе update (как bias segment actuals): k ∈ {1,2,3}.
2. Для каждого завершённого сегмента `k`:
   - `r30`, `r20` = доля минут с flat-окнами 30m / 20m на срезе сегмента (`_chop_segment_flat_coverage_fraction`);
   - `median20` = медиана столбца `chop_dur_20_k` по **всей** `regression_df` (кэш);
   - `chop_actuals[k] = blend_chop30_with_chop20(r30, r20, median20)` — та же формула притяжения к медиане, что [§4.3](#43-segment-targets-session-chop_dur-and-3020-columns).
3. Результат — доли **0..1** для якорения staged-инференса (`intraday_chop_segment_actuals` в `predict_session_chop`).

### 22.2 Layer A2: effective chop and guardrail

`compute_effective_intraday_chop_segments(chop_result, chop_actuals)`:

1. Для k ≤ завершённых сегментов — **факт** из `chop_actuals`; для остальных — модель `predicted_chop_pct_k / 100`.
2. **Guardrail** (если средний факт завершённых сегментов высок — «сильный chop day»):
   - `realized_mean ≥ 0.70` → для **всех незавершённых** k поднять долю минимум до **0.55** (`floor_55`);
   - `realized_mean ≥ 0.60` → минимум **0.45** (`floor_45`).
3. Сессионный **`effective_chop_pct`** = минутно-взвешенная сумма `f1..f4` (`session_weighted_chop_from_segments`).
4. Возвращает `predicted_chop_seg_pct` (кортеж из 4 процентов после guardrail), diagnostics для `bias_last_prediction_debug.json`.

### 22.3 Reapply EM / targets / zones

`reapply_intraday_chop_session_result(...)` после A2:

1. Записывает `effective_chop_pct` и сегментные проценты в `ChopSessionResult`.
2. Пересчитывает **`get_em_from_chop_calibration`** и **`compute_target_of_session_from_similar_chop`** с:
   - `predicted_chop_seg_pct` (после guardrail);
   - `intraday_pool_anchors` ([§14.2](#142-intraday-session_range-anchors-and-post-filter));
   - теми же `pred_bias`, `pred_pm_*`, что утренний прогон.
3. **`enforce_chop_targets_cover_expected_range`** — targets не уже ER ([§16](#16-scenario-targets)).
4. Опционально обновляет chop-zones по новым `w_bull` / `w_bear`.

Параметры `predict_session_chop` для update (из `update_TradePlan`):

- `intraday_chop_segment_actuals` — факты или `{}` если blended actuals недоступны;
- `intraday_predict_stages` — `[2,3,4]`, `[3,4]` или `[4]` по фазе; при отсутствии якорей — полная цепочка `[1,2,3,4]`;
- `spx_1m` / `spx_1m_for_regression_cache` — merged поток (история + `SPX_last_data.csv`).

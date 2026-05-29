# Модуль `bias.py`: регрессионная таблица, таргеты, отбор признаков и многостадийное предсказание

Документ описывает **текущую** реализацию [`bias.py`](../bias.py). При расхождении с кодом источником истины является сам модуль.

## 1. Цели модуля

- **Прогноз направленности и силы сессионного движения SPX** на целевой торговый день **D** по набору признаков, построенных из завершённых сессий **D−1…D−4**, премаркета **D** и рыночного контекста (ES, VIX).
- **Две согласованные шкалы**:
  - **direction** — оценка направления (знак/режим дня в компактной форме [−1, 1]);
  - **strength** — оценка силы движения с учётом эффективности пути и масштаба относительно ATR.
- **Четыре внутридневных интервала RTH** (минутные сегменты, см. §6.2): для каждой шкалы **direction** и **strength** обучаются и применяются **четыре стадии** моделей (таргеты `bias_dir_k` / `bias_strength_k`, *k* = 1…4). На стадии 2 в признаки добавляется итог стадии 1 **той же цепочки** (для `dir` — `bias_dir_1`, для `strength` — `bias_strength_1`); на стадии 3 — **среднее** сырьевых таргетов сегментов 1 и 2; на стадии 4 — **среднее** сегментов 1–3 (см. §6.3 и детальный разбор §12).
- **Итоговые скаляры для смежных модулей** (например `chop`, TradePlan):
  - **`bias_dir_for_chop`** — взвешенное по длительности сегментов (минуты) объединение **четырёх** предсказанных значений **`bias_dir_1`–`bias_dir_4`**. Каждое число берётся из поля **`y_bias`** результата **`_apply_bias_model_to_features`** для соответствующей стадии и цепочки **dir**, т.е. это **primary-калиброванный** выход Ridge (не «сырой» dot-product до калибровки и не то же самое, что **`y_bias_raw`** в диагностике).
  - **`bias_strength_for_chop`** — то же для цепочки **strength** (`bias_strength_1`–`bias_strength_4` из primary **`y_bias`** стадий `strength`).
  - **`bias_for_chop`** и публичное поле **`y_bias`** в ответе **`predict_bias_and_probability`** — **та же формула**, что у **`bias_combined_k`**: **`bias_segment_combined_from_raw`** с весами **`BIAS_COMBINED_*`**, но вход — **`bias_dir_for_chop`** и **`bias_strength_for_chop`** (минутные chop-агрегаты из §12.3), а квантили для **`bias_raw_to_unified_score`** берутся из JSON **стадии 4** для `dir` и `strength` (согласовано с текстовыми зонами chop для этих двух скаляров в консоли). **`bias_combined_1`–`bias_combined_4`** по-прежнему считаются **посегментно** по парам \((\hat{b}^{dir}_k,\hat{b}^{str}_k)\) и квантилям стадии **k**.

Легаси: среднее **body_norm** по всей RTH-сессии (`_y_bias_session`) в общую регрессионную таблицу **не пишется**; остаётся вспомогательной функцией в коде.

### 1.1 Сквозная последовательность (от файлов до полей ответа)

Ниже — порядок, в котором код фактически связывает данные с числами, чтобы по одному описанию можно было восстановить логику `bias.py`.

1. **Сырые CSV** в **`data_spx`** (SPX 1m как источник календаря RTH и баров таргета; ES премаркет/RTH; VIX/VIX9D и т.д.) читаются при построении таблицы или при точечном вычислении признаков на дату **D**.
2. **`build_bias_regression_table` / инкремент** формируют строки **`predict_date = D`** с признаками по **D−1…D−4** и таргетами на **полный день D** и по **четырём сегментам D** (`y_bias_*`, `bias_*_k`). Опционально дата исключается из кэша (§5.0).
3. **Offline (два раза)** — отбор признаков и α по **`y_bias_dir`** и по **`y_bias_strength`** → два JSON hyperparams §4.2; затем при полном офлайн-контуре переобучаются **pooled** модели полной сессии.
4. **Ежедневно или в финале офлайн** выполняется **`train_predict_bias_segments`**: для **каждой** стадии 1→4 внутри стадии сначала **dir**, затем **strength** переобучается Ridge на истории с тем же стадированным видом признаков, что и при таргете; на **прогнозный день** в словарь признаков поочерёдно подставляются уже полученные **primary** **`y_bias`** предыдущих стадий **той же цепочки** (§6.3). Режим primary-калибровки для обучения и применения выбирается **отдельно** для `dir` и `strength` (**`_bias_primary_calibration_mode_for_target`** → `BIAS_DIRECTION_PRIMARY_CALIBRATION_MODE` / `BIAS_STRENGTH_PRIMARY_CALIBRATION_MODE`).
5. Из восьми наборов выходов берутся числа **`bias_dir_k`**, **`bias_strength_k`** (везде калиброванный **`y_bias`**); по ним считаются **`bias_*_for_chop`** (взвешивание минутами) и **`bias_combined_*`**, **`bias_for_chop`** (§12.3–12.4).
6. **`predict_bias_and_probability`** подменяет публичный **`y_bias`** на **`bias_for_chop`**, а текстовые ярлыки берёт из **стадии 4 direction** (§13); при **`save_audit`** пишется JSON из полей **`audit_payload`** в **`train_predict_bias_segments`**.

---

## 2. Обзор архитектуры

```text
Источники CSV (data_spx)
        ↓
build_bias_regression_table / update_bias_regression_table_cache_incremental
        ↓
Таблица: predict_date D | признаки (D−1…D−4, premarket D, …)
        | y_bias_dir, y_bias_strength (полная сессия D)
        | bias_dir_k, bias_strength_k (k=1..4)
        ↓
┌───────────────────────────────┬───────────────────────────────┐
│ Offline (RUN_OFFLINE_         │ Daily (RUN_OFFLINE_           │
│ SELECTION=True)               │ SELECTION=False)              │
├───────────────────────────────┼───────────────────────────────┤
│ 2× _run_offline_selection:    │ Загрузка hp_dir + hp_strength │
│  • y_bias_dir  → hp_dir.json  │ Инкремент / обновление кэша   │
│  • y_bias_strength → hp_str…  │   таблицы (см. исключение даты│
│ train_bias_model полный train │   plan-for в коде)             │
│  → pooled JSON (dir/str)      │ train_bias_model на train-    │
│ train_predict_bias_segments   │   подмножестве: y_bias_dir /  │
│  → 8× staged JSON (4+4) +     │   y_bias_strength → pooled    │
│     audit                     │   JSON                        │
│                               │ train_predict_bias_segments   │
│                               │  → 8× staged JSON + audit     │
└───────────────────────────────┴───────────────────────────────┘
```

**Производственный прогноз**: `predict_bias_and_probability()` вызывает `train_predict_bias_segments()`. Публичное поле **`y_bias`** совпадает с **`bias_for_chop`**: combined на **`bias_dir_for_chop`** и **`bias_strength_for_chop`** с квантилями стадии 4 (§12.4). Текстовые **`primary_bias`** и **`bias_strength`** берутся из диагностики **последней стадии direction** (`bias_dir_4`). Числовые поля **`bias_dir_for_chop`**, **`bias_strength_for_chop`**, **`bias_for_chop`**, **`bias_combined_1..4`** выдаются отдельно; подробности — §12.

---

## 3. Календарь и временные окна

| Понятие | Значение в коде |
|--------|------------------|
| **D** | `predict_date` строки таблицы |
| **Торговый календарь** | Уникальные даты RTH по SPX 1m (`SPX_1min_90d_ibkr.csv`), маска **09:30–15:59** NY |
| **Первая целевая строка** | `FIRST_TARGET_SESSION_INDEX = 4`: первая **D** = `spx_dates[4]` (пять завершённых сессий в индексах 0–4 включительно, чтобы история содержала D−1…D−4); в коде сборки строк цикл идёт с `for i in range(4, len(trading_dates))` |
| **RTH для SPX 1m** | `RTH_START=09:30`, `RTH_END=15:59` |
| **RTH до 16:00** | `RTH_END_16` — для 15m SPX / VIX 5m |
| **Длина сессии в минутах** | `RTH_MINUTES_PER_SESSION = 390` |

---

## 4. Источники данных и артефакты

### 4.1 Каталог данных

`DATA_DIR = bias.py → data_spx/`

Типичные файлы:

- SPX 1m RTH, SPX daily, SPX 15m RTH  
- ES 1m (премаркет), ES 1m RTH  
- VIX / VIX9D 5m  

### 4.2 Файлы в корне проекта (рядом с `bias.py`)

| Файл | Назначение |
|------|------------|
| `bias_regression_table.csv.gz` | Кэш регрессионной таблицы |
| `bias_regression_hyperparams_dir.json` | Результат offline-отбора по **`y_bias_dir`** |
| `bias_regression_hyperparams_strength.json` | Результат offline-отбора по **`y_bias_strength`** |
| `bias_regression_model.json` | Итоговая Ridge+калибровки на **полной сессии direction** (`y_bias_dir`) после offline |
| `bias_regression_model_strength.json` | То же для **полной сессии strength** (`y_bias_strength`) |
| `bias_regression_model_dir_1.json` … `bias_regression_model_dir_4.json` | Стадии направления (**4** файла); имя kind в пути — `dir` |
| `bias_regression_model_strength_1.json` … `bias_regression_model_strength_4.json` | Стадии силы (**4** файла); kind — `strength` |
| `bias_last_prediction_debug.json` | Аудит последнего прогона (стадии, chop-веса, α, фичи) |
| `bias_calibration_validation_compare.json` | Сводка `evaluate_model_buckets` при сохранении |

---

## 5. Регрессионная таблица

Строится `build_bias_regression_table()` / дополняется `update_bias_regression_table_cache_incremental()`.

### 5.0 Исключение даты «новой сессии» из таблицы

Параметры **`exclude_predict_session_date`** (`build_bias_regression_table`, `_build_bias_regression_rows_for_dates`, `update_bias_regression_table_cache_incremental`) и логика **`--plan-for`** / `BIAS_CLI_SESSION` в **`if __name__ == "__main__"`** делают одно и то же по смыслу: **строка с `predict_date`, равным этой календарной дате, не попадает в кэш и не участвует как обучающий пример**. Это нужно, чтобы не учить Ridge на незавершённой или «завтрашней» intraday-сессии и не смешивать утечку с прогнозом. При инкрементальном обновлении кэш читается и при необходимости **фильтруется** (строка с исключённой датой удаляется). Прогноз для этой даты строится только через **`train_predict_bias_segments`** по правилам §12 — отдельно от строки таблицы.

### 5.1 Строка таблицы и отбор строк

- **Одна строка** = одна целевая дата **D** (`predict_date`).
- **Признаки** = результат `get_bias_regression_features(...)` (история D−1…D−4, премаркет D, IV и т.д.).
- Строка отбрасывается, если для D нет блока ES premarket (00:00–09:30), когда ES загружен.

### 5.2 Колонки таргетов и сегментов

В **сохранённый** gzip-кэш (`bias_regression_table.csv.gz`) пишутся:

- **`y_bias_dir`**, **`y_bias_strength`** — таргеты на **полную RTH-сессию D** (см. §6).
- **`bias_dir_1`…`bias_dir_4`**, **`bias_strength_1`…`bias_strength_4`** — таргеты по **четырём** последовательным минутным сегментам RTH одного и того же дня D (см. §6.2; порядок среза `iloc` — в коде `_bias_segment_targets_from_block`).

Производные столбцы **`bias_*_1_2_mean`** и **`bias_*_1_3_mean`** в общий файл **не записываются**: они считаются **только во временном** **`stage_df`** внутри **`_bias_segment_table_from_base`** при обучении стадий 3–4 (§6.3).

Фикс **`BIAS_TARGET_AND_SEGMENT_COLUMNS`** включает и синтетическое **`"y_bias"`**, и производные средние (`bias_dir_1_2_mean` и т.д.) — они **исключаются из корреляционного сканирования признаков** как зависимости таргета, даже когда появляются в оперативных датафреймах staging.

### 5.3 Где фигурирует имя `y_bias` (это не «третий таргет таблицы»)

В **CSV/кэше регрессионной таблицы** колонки **`y_bias` нет**. Там только **`y_bias_dir`**, **`y_bias_strength`** и сегменты **`bias_dir_k` / `bias_strength_k`**.

Имя **`y_bias`** подставляется в коде в трёх разных ролях:

1. **`train_bias_model(..., target_y_column=...)`**  
   Функция всегда читает целевую колонку из переданного имени. Для pooled-моделей это **`y_bias_dir`** или **`y_bias_strength`**. Для **каждой стадии** в `train_predict_bias_segments` передаётся **`target_y_column="y_bias"`**, потому что перед этим строится временный датафрейм в **`_bias_segment_table_from_base`**, где в колонку **`y_bias`** копируют **ровно один** сегментный таргет того этапа — `bias_dir_k` или `bias_strength_k` при *k* ∈ {1,2,3,4} (см. `target_col`). То есть **`y_bias` здесь — универсальное имя столбца «текущий таргет одной стадии»**, а не отдельная экономическая величина.

2. **`_run_offline_selection_and_save`**  
   Отбор признаков делается **дважды** по полносессионным колонкам **`y_bias_dir`** и **`y_bias_strength`**. Внутри процедуры делается копия: `df["y_bias"] = df[target_y_column]` — это **временный псевдоним** для единообразного кода (обучение Ridge, квантили, метрики), чтобы не дублировать ветки под разные имена колонок.

3. **Словари предсказания и API**  
   В **`_apply_bias_model_to_features`** в результате ключ **`"y_bias"`** — это **уже откалиброванный итоговый score** модели для **данной стадии** (primary). В **`predict_bias_and_probability`** верхнеуровневый **`y_bias`** — это **`bias_for_chop`**: **`bias_segment_combined_from_raw`** от **`bias_dir_for_chop`** и **`bias_strength_for_chop`** с квантилями стадии 4 (см. §12.4), а **не** столбец таблицы и **не** то же самое, что только **`bias_dir_for_chop`** или только **`bias_strength_for_chop`**.

Итого: **offline selection и pooled train действительно завязаны на `y_bias_dir` и `y_bias_strength`**. **`y_bias` как имя колонки** нужно **стадийному** обучению и **внутренним копиям** при отборе; **в финальной таблице** его нет.

---

## 6. Таргеты: формулы

Обозначения для минутного блока сессии: Open/High/Low/Close; \(\varepsilon =\) `BODY_NORM_EPS` где нужно.

### 6.1 Полная сессия D: `y_bias_dir` и `y_bias_strength`

На **полном RTH-блоке** дня D (`_get_spx_rth_session` + отсортированные по времени бары):

1. **Диапазон сессии**  
   \(\text{session\_range} = \max(High) - \min(Low)\) по барам.

2. **Направленное движение**  
   \(\text{directional\_move} = Close_{\text{last}} - Open_{\text{first}}\).

3. **Direction (таргет dir)**  
   \[
   y_{\text{dir}} = \mathrm{clip}\left(\frac{\text{directional\_move}}{\text{session\_range} + \varepsilon}, -1, 1\right)
   \]

4. **Efficiency**  
   \(\text{path} = \sum_t |Close_t - Close_{t-1}|\),  
   \(\text{efficiency} = |\text{directional\_move}| / (\text{path} + \varepsilon)\).

5. **ATR14 дня D**  
   Берётся из дневного ряда (`atr_14` на дате D): `_daily_atr14_for_date`.

6. **Strength (таргет strength)**  
   \(\text{atr\_norm} = \text{session\_range} / ATR_{14}\) (если ATR неконечен — используется 1.0 в коде).  
   \[
   y_{\text{strength}} = \mathrm{clip}\left( y_{\text{dir}} \cdot \sqrt{\max(\text{atr\_norm}, 0)} \cdot \text{efficiency}, -1, 1 \right)
   \]

Именно эти величины записываются в **`y_bias_dir`** и **`y_bias_strength`**.

### 6.2 Четыре сегмента RTH по минутным барам (фиксированные длины)

Сумма длин сегментов = **`RTH_MINUTES_PER_SESSION` = 390** (соответствие широкому RTH-блоку 09:30–close в минутках). Комментарий в коде привязан к календарным окнам **ET** как ориентир:

| Сегмент | Длительность (мин.) | Константа | Комментарий в коде (ориентир) |
|--------|----------------------|-----------|-------------------------------|
| 1 | **90** | `BIAS_SEGMENT_1_MINUTES` | первый блок после открытия |
| 2 | **90** | `BIAS_SEGMENT_2_MINUTES` | следующий блок |
| 3 | **90** | `BIAS_SEGMENT_3_MINUTES` | следующий блок |
| 4 | **120** (= 390 − 90×3) | `BIAS_SEGMENT_4_MINUTES` | хвост сессии до close |

Разбиение задаётся **числом минут подряд** в отсортированном RTH-фрейме дня (**не** метками «12:00» в документе): `e1`, `e1+e2`, `e1+e2+e3`, затем остаток — см. **`_bias_segment_targets_from_block`**.

На **каждом** таком интервале вычисляется пара **`(bias_dir_k, bias_strength_k)`** той же идеей, что и полная сессия в §6.1 (**range**, **directional_move**, затем формулы dir и strength), но на подвыборке баров этого сегмента и с **ATR14 дня D** (`_daily_atr14_for_date`).

Пример временных надписей в консоли/утилите форматирования стадий (для человека): см. **`format_bias_segments_console`** (`09:30–11:00`, `11:00–12:30`, …) — они **служат подписью** и должны считаться следствием минутных длин, если данные без пропусков.

### 6.3 Признак X vs таргет y на стадиях `_bias_segment_table_from_base`

Для каждого **`target_kind ∈ {dir, strength}`** строится свой набор признаков (списки коэффициентов задаются двумя JSON hyperparams для direction и strength; см. **`train_predict_bias_segments`**).

| Стадия | Колонка таргета ``y_bias`` (= копия сегментной колонки) | Имя сегментного таргета в CSV-кэше | Дополнительная колонка в матрице X (поверх сохранённого списка фич заказа регрессии) | Как заполняется **в исторических строках `stage_df`** при обучении (`_bias_segment_table_from_base`; фактические `bias_*_k` читаются из общего регрессионного кэша) | Что подставляется **при прогнозе нового дня** в словарь `features_for_stage` (`train_predict_bias_segments`) |
|--------|----------------------------|------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1 | `bias_dir_1` или `bias_strength_1` | уже есть в CSV | нет | — | базовые признаки только из **`_bias_prepare_current_features_for_segments`** (нет сегментных `bias_*` из прошлых стадий) |
| 2 | `*_2` | `bias_*_2` в CSV | **`bias_dir_1`** или **`bias_strength_1`** | Фактический **`bias_*_1`** из той же строки истории дня в таблице | **Откалиброванное** значение после предсказания стадии 1 (**только эта же цепочка** dir или strength): `stage_outputs[target_kind][1]["y_bias"]` |
| 3 | `*_3` | `bias_*_3` | **`bias_dir_1_2_mean`** / **`bias_strength_1_2_mean`** | **Среднее** фактических `bias_*_1` и `*_2` в строке истории (**0.5·(v1+v2)**) | **Среднее** уже предсказанных на этом же прогоне \(\hat{b}_1\) и \(\hat{b}_2\) этой цепочки (если оба конечны) |
| 4 | `*_4` | `bias_*_4` | **`bias_dir_1_3_mean`** / **`bias_strength_1_3_mean`** | **Среднее** фактических `bias_*_1`, `*_2`, `*_3` (**(v1+v2+v3)/3**) | **Среднее** уже предсказанных \(\hat{b}_1,\hat{b}_2,\hat{b}_3\) этой цепочки |

**Важное различие:** при **обучении** строки истории уже содержат **факт** будущего сегмента в столбце `bias_*_k` (этот столбец и копируется в `y_bias` временного staging-фрейма). При **прогнозе**, если день не в таблице, сегментные столбцы **не читаются** из CSV; вместо этого в **extra** попадают **только что посчитанные** моделью числа того же номера цепочки.

**Отдельно:** ответ пайплайна и аудит дублируют числовые поля именами таргета (`bias_dir_k`, `bias_strength_k`), но **значение** там — ключ **`y_bias`** из результата **`_apply_bias_model_to_features`**, то есть **primary после калибровки**. На типичном overnight-прогнозе они **не** читаются из ячеек строки **`predict_date = D`** в кэше: для этого дня сначала считаются только базовые признаки (§12.1), затем по очереди рождаются калиброванные \(\hat{b}\). Сырой выход регрессии доступен во внутреннем поле **`y_bias_raw`** (и сопутствующие варианты калибровки) в каждой стадии, но **`bias_*_for_chop`** и построение **`bias_combined_*`** опираются на **primary** (`y_bias`).

Подстановка **фактических** `bias_*_1`, `bias_*_2`, … из завершённых intraday-баров вместо модельных \(\hat{b}\) возможна только во внешних сценариях; «из коробки» **`train_predict_bias_segments`** так не делает.

---

## 7. Признаки (`REGRESSION_FEATURE_ORDER`)

Реализация: **`get_bias_regression_features`** ([`bias.py`](../bias.py)); для прогнозной даты **D** на вход подаются календарные сессии **D−1…D−4** (и опционально **D−5** для части вспомогательных рядов), плюс загруженные датафреймы SPX 1m, ES 1m / ES RTH 1m, VIX / VIX9D 5m, дневной SPX, 15m SPX.

### 7.0 Общие правила

- **Режим «без утечки intraday D»**: при **`exclude_spx_rth_on_date`** из минутного SPX, 15m SPX и дневного ряда **исключается RTH целевой сессии** (`bias_exclude_forecast_session_rth`, `bias_trim_spx_daily_before`, `bias_trim_spx_15m_forecast_session`). **Премаркет ES** и **VIX на D** **не** режутся (см. докстринг `get_bias_regression_features`).
- **SPX 1m для признаков** после возможного тримминга обозначается в коде как **`spx_feats`**; сессия RTH дня `x` = **`_get_spx_rth_session(spx_feats, x)`** (маска **09:30–15:59** NY, `RTH_START` / `RTH_END`).
- **ES RTH** — только **`ES_1min_90d_RTH.csv`** (`es_rth_1m`), маска та же.
- **Премаркет ES** — **`_es_overnight_premarket_block`**: **[00:00, 09:30)** календарного дня (константа **`OVERNIGHT_START`**, граница **`RTH_START`**).
- **Стандартный «gap»-паттерн** для серии \(f(d)\) по минутным SPX-блокам:  
  \(\text{gap}_f = f(D{-}1) - \mathrm{mean}\bigl(f(D{-}2), f(D{-}3), f(D{-}4)\bigr)\), где в среднее попадают только конечные значения; иначе признак **NaN**. Для части колонок используется **z-подобная** нормировка — см. **`_iv_zscore_from_prev`** и отдельные формулы ниже.
- **Константы порогов** (централизованно в `bias.py`):  
  `BODY_NORM_EPS`, `LOG_EPS`, `SMALL_RANGE_NORM_THRESHOLD` (0.7), `BIG_RANGE_NORM_THRESHOLD` (1.5), `CHOP_INDEX_WINDOW` (14), `CHOP_INIT_BARS` (15 → усреднение chop с **16-й** минуты, индекс среза `iloc[CHOP_INIT_BARS:]`), `IV_ZSCORE_STD_FLOOR`, `IV_ZSCORE_CLIP`.
- **Порядок колонок** в CSV и в списках отбора строго **`REGRESSION_FEATURE_ORDER`** (ниже — в том же порядке).

### 7.1 Исключения из модели

`FEATURES_EXCLUDED_FROM_MODEL` — колонки **присутствуют в таблице**, но **не входят в X** Ridge и не в стартовый пул offline-отбора:

- `close_5d_gap`
- `vwap_mean_ratio`
- `downside_tail_ratio_gap`
- `mean_dist_vwap_gap`

Стартовое число признаков в модели: **len(REGRESSION_FEATURE_ORDER) − 4 = 21**.

---

### 7.2 Поштучная спецификация (как в коде, для правок формул)

Ниже для каждого имени: **смысл**, **откуда данные**, **точная формула / вызовы**, **что менять при улучшении**.

#### `body_norm_zscore_spx`

- **Смысл**: насколько средний «нормированный корпус» минутных свечей на **D−1** отклоняется от недавней базы **D−2…D−4** в единицах их разброса.
- **Данные**: SPX 1m RTH, блоки `block_1`…`block_4` (после `spx_feats`).
- **База на день** \(d\): **`_body_norm_mean_session`**: по всем барам сессии \(\text{mean}\bigl((C-O)/(H-L+\varepsilon)\bigr)\), \(\varepsilon=\) **`BODY_NORM_EPS`**.
- **Признак**:  
  \(\displaystyle z = \frac{b_1 - \mathrm{mean}(b_2,b_3,b_4)}{\mathrm{std}(b_2,b_3,b_4) + \mathrm{BODY\_NORM\_EPS}}\)  
  при конечном \(b_1\) и хотя бы одном конечном из \(b_2,b_3,b_4\); результат **округляется** до 6 знаков (`round(..., 6)`). Иначе **NaN**.

#### `loc_5d_spx`

- **Смысл**: где закрылся **D−1** внутри **четырёхдневного** high/low окна **D−4…D−1** (дневной SPX).
- **Данные**: `spx_daily_feat` (после тримминга), список **`d_list_5[:4]`** = ровно эти четыре даты.
- **Формула**:  
  \(H=\max High\), \(L=\min Low\) по четырём дням;  
  \(\text{loc} = \mathrm{clip}\bigl((Close_{D-1}-L)/(H-L+\mathrm{BODY\_NORM\_EPS}),\,0,\,1\bigr)\).

#### `chop_mean_gap`

- **Смысл**: сдвиг «режущего» индекса относительно недавнего среднего.
- **Данные**: SPX 1m RTH.
- **По дню** \(d\): **`_chop_mean_session`**: классический **Chop Index** по TR, HH\(_{14}\), LL\(_{14}\) с **`CHOP_INDEX_WINDOW`**, **`shift(1)`** на индикаторах; `chop_index = 100·log10(sum_TR/(HH-LL))/log10(14)`; берётся **среднее** по `chop_index` для баров с индекса **`CHOP_INIT_BARS`** до конца (минимум **16** баров в сессии). Требуются колонки High/Low/Close.
- **Признак**: **`round(chop_mean(D−1) − mean(chop_mean(D−2..D−4)), 4)`**.

#### `premarket_close_position`

- **Смысл**: насколько закрытие ES в премаркете **D** отличается от среднего премаркет-профиля **D−1…D−3**.
- **Данные**: ES 1m, окно **[00:00, 09:30)** на каждом дне.
- **По дню** \(d\): **`_premarket_close_position_day`**: \(pos = (Close_{last}-L)/(H-L+\mathrm{LOG\_EPS})\), **clip** в \([0,1]\); High/Low по всему премаркет-блоку, **последний** Close — последний бар до RTH.
- **Валидация**: **`_es_premarket_dates_match_spx`** для множества `{D, D−1, D−2, D−3}`; иначе **NaN**.
- **Признак**: **`round(pos(D) − mean(pos(D−1), pos(D−2), pos(D−3)), 4)`** (среднее только по конечным).

#### `small_candles_gap`

- **Смысл**: дисбаланс **малых** баров (низкий range относительно intraday ATR14) с раздельным учётом бычьих/медвежьих тел.
- **Данные**: SPX 1m RTH.
- **По дню**: **`_small_range_ratio_session`**: требуется **≥16** баров; **`range_norm = (H-L)/(ATR14_i+LOG_EPS)`** по массиву **`_atr14_per_bar`**; знаменатель **N** = число баров с конечными `range_norm`, O, C; малые бары: `range_norm ≤ SMALL_RANGE_NORM_THRESHOLD`;  
  \(f = \#\text{малых up}/N - \#\text{малых down}/N\), затем **`_round_ratio`** (3 знака).
- **Признак**: **`round(f(D−1) − mean(f(D−2..D−4)), 4)`**.

#### `big_candles_gap`

- **Смысл**: то же для **крупных** баров (`range_norm ≥ BIG_RANGE_NORM_THRESHOLD`).
- **Реализация**: **`_big_range_ratio_session`**, далее как у `small_candles_gap`.

#### `close_5d_gap`

- **Данные**: дневной SPX (`spx_daily_feat`), ровно четыре даты **D−4, D−3, D−2, D−1** (как у `loc_5d_spx`). На каждой дате нужны **Open, High, Low, Close**.

- **Шаг A — полоса и знаменатель (никаких отдельных имён вроде «denom» в коде нет, это просто выражение):**  
  - \(H = \max(High_{D-4},\, High_{D-3},\, High_{D-2},\, High_{D-1})\).  
  - \(L = \min(Low_{D-4},\, Low_{D-3},\, Low_{D-2},\, Low_{D-1})\).  
  - \(\varepsilon\) — константа **`BODY_NORM_EPS`** из `bias.py`.  
  **Один общий знаменатель для всех четырёх дробей ниже:**
  \[
  H - L + \varepsilon.
  \]
  В коде: `(max(High) - min(Low)) + BODY_NORM_EPS` по тем же четырём дням. Должно быть **> 0**, иначе признак **NaN**.

- **Шаг B — четыре значения одной серии (одна полоса, одно открытие якоря \(Open_{D-4}\), один знаменатель \(H-L+\varepsilon\)):**  
  Числитель всегда: «**Close этого дня минус открытие первого дня полосы (D−4)**».

  \[
  r_{D-4} = \frac{Close_{D-4} - Open_{D-4}}{H - L + \varepsilon}
  \]

  \[
  r_{D-3} = \frac{Close_{D-3} - Open_{D-4}}{H - L + \varepsilon}
  \]

  \[
  r_{D-2} = \frac{Close_{D-2} - Open_{D-4}}{H - L + \varepsilon}
  \]

  \[
  r_{D-1} = \frac{Close_{D-1} - Open_{D-4}}{H - L + \varepsilon}
  \]

  Последняя строка — это **та самая «сырая» накопленная нормировка** от **\(Open_{D-4}\)** до **\(Close_{D-1}\)** внутри четырёхдневной полосы (как в вашем первом сообщении к треду). **Это не** формула «только \(Close_{D-4}-Open_{D-4}\)» — та фигурирует лишь в \(r_{D-4}\).

- **Шаг C — gap (колонка `close_5d_gap`):** отклонение **\(r_{D-1}\)** от среднего **\(r_{D-2}, r_{D-3}, r_{D-4}\)** (в среднее входят только конечные значения, как в коде — не обязательно ровно деление на 3):
  \[
  close\_5d\_gap = r_{D-1} - \mathrm{mean}(r_{D-2},\, r_{D-3},\, r_{D-4}).
  \]
  Итог в коде: **`round(..., 4)`**; при отсутствии любой из четырёх дневок или при \(H-L+\varepsilon \le 0\) — **NaN**.

- **Та же логика одной строкой без LaTeX** (удобно, если формулы не рендерятся):

```text
Знаменатель = max(High на D-4, D-3, D-2, D-1) - min(Low на тех же днях) + BODY_NORM_EPS
r_d       = (Close_d - Open_D-4) / Знаменатель   для d = D-4, D-3, D-2, D-1
close_5d_gap = r_D-1 - mean(только конечные из r_D-2, r_D-3, r_D-4)
```

- **Если ваш просмотрщик Markdown криво рисует формулы** (например, показывает «aligned» или `[0.35em]` текстом): ориентируйтесь на **шаг A** (явный знаменатель \(H-L+\varepsilon\)) и на **шаг B** — четыре отдельные однострочные дроби без таблиц.

- **Модель**: колонка в **`FEATURES_EXCLUDED_FROM_MODEL`**.

#### `efficiency_gap`

- **Смысл**: насколько день **D−1** ближе к «идеальному тренду» относительно базы по **D−2…D−4**.
- **По дню**: **`_efficiency_session`**: \(\text{eff} = (C_{last}-O_{first})/(\sum_{t}|ΔC_t| + \mathrm{LOG\_EPS})\), минимум 2 бара.
- **Признак**: **`round(eff(D−1) − mean(eff(D−2..D−4)), 4)`**.

#### `direction_run_gap`

- **Смысл**: пульсации **серий** знака тела свечи (внутри однонаправленных серий минутной структуры).
- **По дню**: **`_direction_run_ratio_session`**: из знаков \(\mathrm{sign}(C-O)\) отбрасываются нули и нечисловые; **longest run** одинакового знака; значение = **longest / N\_valid**.
- **Признак**: **разность без округления**: \(run(D−1) - mean(run(D−2..D−4))\).

#### `entropy_gap`

- **Смысл**: насколько внутридневное распределение знака тела **упорядочено** или «шумно» (энтропия × доминирующая сторона).
- **По дню**: **`_entropy_ratio_session`**: только \(C≠O\); \(p_{up}, p_{down}\); Shannon \(h=-(p\ln p)/\ln 2\) по двум состояниям; **signed_entropy** = \((p_{up}-p_{down})\cdot entropy\_ratio\) (ветка \(\ln\) с **`LOG_EPS`** стабилизации).
- **Признак**: \(ent(D−1) - mean(ent(D−2..D−4))\).

#### `one_sidedness_gap`

- **Смысл**: «вынос» закрытия от открытия относительно **сессионного** диапазона (односторонность дня).
- **По дню**: **`_one_sidedness_session`**: \(|C_{last}-O_{first}|/(H_{sess,max}-L_{sess,min}+LOG\_EPS)\).
- **Признак**: \(os(D−1) - mean(os(D−2..D−4))\).

#### `directional_range_regime`

- **Смысл**: величина дневного трендового компонента **D−1** к **типичному** недавнему диапазону минутной сессии.
- **По дням для базы диапазона**: **`_range_session`** на **SPX**: \(High_{max}-Low_{min}\) за день **D−2,D−3,D−4**.
- **Признак**: \((C_{last}-O_{first})_{D-1}\big/\big(mean(range_{D-2},range_{D-3},range_{D-4}) + LOG\_EPS)\). **Не** использует паттерн «gap от среднего f» — одна безразмерная величина.

#### `mean_dist_vwap_gap`

- **Смысл**: изменение среднего **подписанного** расстояния цены закрытия ES от **intraday VWAP** (не модуля), нормированного на **`VWAP+ε`**.
- **Данные**: **ES RTH 1m** (не SPX): сессионный VWAP строго от **.Volume > 0** (`_es_vwap_series`); иначе **NaN** для дня.
- **По дню**: **`_mean_dist_vwap_session_es`**: по всем минутным барам с конечными Close и VWAP:  
  **`mean((Close − VWAP) / (VWAP + LOG_EPS))`** (**не** деление на ATR; **со знаком** — в отличие от SPX-варианта `_mean_dist_vwap_session` с модулем, который здесь не используется).
- **Признак**: \(mdv(D−1) - mean(mdv(D−2..D−4))\).
- **Модель**: **`FEATURES_EXCLUDED_FROM_MODEL`**.

#### `vwap_cross_rate_gap`

- **Смысл**: частота «качелей» около VWAP относительно недавней нормы.
- **Данные**: ES RTH 1m.
- **По дню**: **`_vwap_cross_rate_session_es`**: `sign(C−VWAP)`, выкидываются нули; число смен знака между **последовательными** ненулевыми знаками, делится на **m−1** (m — длина сжатой последовательности знаков).
- **Признак**: \(vcr(D−1) - mean(vcr(D−2..D−4))\).

#### `downside_tail_ratio_gap`

- **Смысл**: доля минутной волатильности, приходящейся на **отрицательные** возвраты закрытия.
- **Данные**: SPX 1m RTH.
- **По дню**: **`_downside_tail_ratio_session`**: \(r_i = C_i-C_{i-1}\); \(\sum |\min(r_i,0)| / (\sum |r_i| + LOG\_EPS)\).
- **Признак**: \(dtr(D−1) - mean(dtr(D−2..D−4))\).
- **Модель**: **`FEATURES_EXCLUDED_FROM_MODEL`**.

#### `mean_y_bias_5`

- **Смысл**: средний нормированный дневной «корпус» за последние **четыре** завершённые сессии перед **D** (имя историческое: «_5», фактически **4** дня в коде докстринга).

- **Данные**: дневной OHLC **`d_list_5[:4]`**.
- **Формула**: для каждого дня \(\frac{C-O}{H-L+\mathrm{BODY\_NORM\_EPS}}\), затем **mean** доступных \(\ge1\) точек → **clip** в \([-1,1]\). **Не gap**.

#### `ema100_gap`

- **Смысл**: изменение среднего отклонения цены от **внутридневной EMA(100)** в единицах **минутного ATR14** на **D−1** против базы по **D−2…D−4**.
- **Данные**: дневной SPX (**`ema_100`** историческое до начала окна из **дат младше \(d\text{-}minus\text{-}4\)** если он есть **иначе \(d\text{-}minus\text{-}2`**), затем «прокачка» EMA только по последовательным барам **четырёх** календарных дней (**`alpha_ema = 0.0198`**, см. литералы внутри функции — **единственное место** задания длины эффективной памяти интрадей EMA для этого признака).
- **По каждому дню множества D−4 … D−1**: столбец `ema100_intraday` добавляется в объединённые минутные бары; затем **`_ema100_distance_session`**: нужно \(\ge16\) минутных баров;  
  \(\text{mean}_i \bigl((C_i-EMA_i)/(ATR14_i+LOG\_EPS)\bigr)\) по барам с конечными значениями.
- **Признак**: **`round(dist(D−1) − mean(dist(D−2..D−4)), 4)`**.

#### `vwap_gap_es`

- **Смысл**: сдвиг типичной **премии VWAP над дневным открытием** в \(ATR\) единицах (структурирование первой части RTH на ES).

- **Данные**: ES RTH 1m.
- **По дню**: **`_vwap_open_distance_session`**: нужно \(\ge16\) баров после `dropna` по OHLCVol; **`open_d = Open`** первого бара; \(\text{mean}_i \bigl((VWAP_i - open\_d)/(ATR14_i + LOG\_EPS)\bigr)\) с **`_atr14_per_bar`** на том же блоке.
- **Признак**: **`round(val(D−1) − mean(val(D−2..D−4)), 4)`**.

#### `vwap_mean_ratio`

- **Смысл**: изменение доли минут, где **`Close > VWAP`**, по ES RTH (**интрабарное участие над/под VWAP**).

- **По дню**: **`_pct_above_vwap_session`**: считает долю среди минут с конечными Close и VWAP.
- **Признак**: **`round(p(D−1) − mean(p(D−2..D−4)), 4)`**.
- **Модель**: **`FEATURES_EXCLUDED_FROM_MODEL`**.

#### `vwap_premarket_distance`

- **Смысл**: положение ES в **премаркете D** относительно **внутреннего** VWAP сегмента, нормированное на **премаркетный** ATR14 по бару (**не gap**).

- **Данные**: ES 1m, **`_es_overnight_premarket_block`**.
- **Формула**: **`_vwap_premarket_mean_dist`** — нужно \(\ge16\) минут после очистки; VWAP объёмный; \(\text{mean}\bigl((C_i-VWAP_i)/(ATR14_i+LOG\_EPS)\bigr)\).

#### `iv_short_long_spread`

- **Смысл**: сколько **краткосрочный «страх» (VIX9D)** дороже/дешевле **VIX** на закрытии RTH по сравнению с недавним режимом.
- **По дню** \(d\): **`_vix_spread_session`**: последний \(Close\) в RTH блоке VIX9D минус последний \(Close\) в RTH блоке **VIX 5m** на дату \(d\) (`_vix_5m_last_rth_close` / строка из `_vix9d_5m_rth_bars`).
- **Признак**: псевдо-z **`_iv_zscore_from_prev`**: «текущий» = **`spread(D−1)`**, «опоры» = \(spread(D{-}2), spread(D{-}3), spread(D{-}4)\) — см. код **`_iv_short_long_spread`** (аргумент `d_minus_5` игнорируется). Никакая дата целевого **D** в формуле не участвует.

#### `iv_overnight_gap`

- **Смысл**: нетипичный **ночной** скачок VIX к началу **D** против предыдущего RTH-закрытия по сравнению с паттерном **трёх** предыдущих ночей.

- **Ключевые вызовы**: **`_overnight_delta_vix(d, prev_rth)`** = \(VIX_{preopen\_last}(d) - VIX_{RTH\_last}(prev\_rth)\) (`_vix_5m_preopen_last_close`, `_vix_5m_last_rth_close`).

- **Признак**: **`_iv_zscore_from_prev( overnight_delta(D), [overnight_delta(D−1), overnight_delta(D−2), overnight_delta(D−3)] )`**. Если **нет \(D−3\) или \(D−4`** в параметрах функции верхнего уровня → **NaN** (строгая проверка **`any(.. is None)`** в **`_iv_overnight_gap`**).

#### `upday_ratio_5_spx`

- **Смысл**: изменение среднего **тела** свечей **15m SPX RTH** между **D−1** и базой по **дневному** паттерну D−2..D−4 (более грубая шкала времени для направления внутри RTH SPX относительно 1m микроструктуры).

- **Данные**: **`_spx_15m_rth_block`**, затем **`_body15_mean_session`** = mean по 15m барам \((C−O)/(H−L+\mathrm{BODY\_NORM\_EPS})\).

- **Признак**: \((bm_{D-1} - mean(bm_{D-2..D-4}))/ (std(bm_{D-2..D-4}) + LOG\_EPS)\). **Клипа по ± `IV_ZSCORE_CLIP` нет** (в отличие от блока `_iv_*`).

#### `iv_short_term_drift`

- **Смысл**: нетипичный **intraday-дрейф** VIX9D в RTH («закупка/перепродажа» кривой вола внутри дня для краткой стороны срока структуры).

- **По дню**: **`_vix9d_drift_rth_session`**: **последний** минус **первый** Close в массиве 5m RTH баров того дня.
- **Признак**: **`_iv_zscore_from_prev(drift(D−1), [drift(D−2), drift(D−3), drift(D−4)])`** (аналогично **`iv_short_long_spread`** по составу истории).

#### `atr14_ratio`

- **Смысл**: относительный уровень **среднеминутного** ATR(14 Wilder по барам) на SPX между **D−1** и **D−2..D−4** (**логарифмический масштаб**).

- **По дню**: **`_atr14_avg_session`**: среднее всех **конечных** элементов массива **`_atr14_per_bar`** (первые ~14 баров — **NaN** в массиве ATR до накопления Wilder-сглаживания, не попадают в mean).
- **Признак**: \(\ln\bigl( (atr_{D-1}+LOG\_EPS)/(mean(atr_{D-2..D-4})+LOG\_EPS) \bigr)\).

---

Функции-строители блоков признаков: помимо **`get_bias_regression_features`**, см. **`_body_norm_mean_session`**, **`_chop_mean_session`**, **`_atr14_per_bar`**, **`_atr14_avg_session`**, **`_small_range_ratio_session`**, **`_big_range_ratio_session`**, **`_efficiency_session`**, **`_direction_run_ratio_session`**, **`_entropy_ratio_session`**, **`_one_sidedness_session`**, **`_range_session`**, **`_mean_dist_vwap_session_es`**, **`_vwap_cross_rate_session_es`**, **`_downside_tail_ratio_session`**, **`_ema100_distance_session`**, **`_vwap_open_distance_session`**, **`_pct_above_vwap_session`**, **`_vwap_premarket_mean_dist`**, **`_iv_short_long_spread`**, **`_iv_overnight_gap`**, **`_upday_ratio_5_spx`**, **`_iv_short_term_drift`**, **`_premarket_close_position_day`**, **`_es_overnight_premarket_block`**.

---

## 8. Управляющие константы (основные)

### 8.1 Режимы и отбор

| Имя | Назначение |
|-----|------------|
| `RUN_OFFLINE_SELECTION` | Дефолт в коде:**`False`** (ежедневный контур с уже сохранёнными hyperparams). `True`: полный rebuild таблицы при `main` + два прогона offline selection + финальный pooled train |
| `N_TEST_SESSIONS` | Число последних **уникальных** дат holdout для selection (**`26`** в текущем `bias.py`) |
| `RIDGE_ALPHA_GRID` | Сетка α для GridSearchCV: **`[20, 30, 40, 50, 60, 70, 80]`** |
| `N_SESSIONS_FOR_HP` | Ограничение горизонта train при GridSearchCV (первые N дат) |
| `OFFLINE_N_TEST_SESSIONS`, `OFFLINE_N_SESSIONS_FOR_HP` | Опциональные переопределения одного прогона |
| `MIN_NEW_SESSIONS_TO_REBUILD` | Порог «новых сессий» для сообщений при сравнении с `selection_last_date` |
| `RIDGE_ALPHA_GRID` | Сетка α Ridge |
| `TIME_SERIES_SPLITS` | Число сплитов `TimeSeriesSplit` |
| `N_FEATURE_SELECTION_RUNS` | Наследие/док-якорь; фактическая процедура — цикл до `MIN_FEATURES_AFTER_SELECTION` |
| `MIN_FEATURES_AFTER_SELECTION` | Нижняя граница числа признаков после pruning (**`12`** в `bias.py`; цикл отбора крутится, пока `len(features_remaining) > MIN_FEATURES_AFTER_SELECTION`) |

### 8.2 Калибровка: переключатели по типу таргета

| Имя | Назначение |
|-----|------------|
| `BIAS_DIRECTION_PRIMARY_CALIBRATION_MODE` | Primary для direction (offline selection с `y_bias_dir`, JSON стадий `dir`) |
| `BIAS_STRENGTH_PRIMARY_CALIBRATION_MODE` | Primary для strength (`y_bias_strength`, стадии `strength`) |
| Допустимые строки | `raw`, `sign_split`, `monotonic_reliability` (или `monotonic` / `reliability`), `hybrid`, `linear` |

Текущие значения по умолчанию в модуле: **`BIAS_DIRECTION_PRIMARY_CALIBRATION_MODE = "linear"`**, **`BIAS_STRENGTH_PRIMARY_CALIBRATION_MODE = "sign_split"`**.

### 8.3 Гибрид и веса

| Имя | Назначение |
|-----|------------|
| `BIAS_HYBRID_SIGN_SPLIT_WEIGHT` | Вес компоненты sign-split в hybrid |
| `BIAS_HYBRID_MONOTONIC_WEIGHT` | Вес monotonic reliability в hybrid |
| `_bias_hybrid_weights()` | Нормализует веса к сумме 1 |

Реализованная смесь: **`y_hybrid = w_sign · y_sign_split + w_mono · y_monotonic`** (с откатом на sign-split, если кривая monotonic выключена или невалидна). Текущие веса в коде: **`BIAS_HYBRID_SIGN_SPLIT_WEIGHT = 0.8`**, **`BIAS_HYBRID_MONOTONIC_WEIGHT = 0.2`**. Комментарии внутри функций могут отличаться — источник истины именно константы.

### 8.4 Sign-split и линейная калибровка

| Имя | Назначение |
|-----|------------|
| `BIAS_CALIB_POST_SHIFT_FACTOR` | Доля post median shift после асимметрии |
| `BIAS_CALIB_K_LAST_N` | Окно последних сессий для оценки k и linear |
| `BIAS_CALIB_K_TRIM_Q` | Признак trim по квантилям `y_true` при median/k (диапазон в процентах) |
| `BIAS_CALIB_K_MIN`, `BIAS_CALIB_K_MAX` | Клампы для \(k_{pos}\), \(k_{neg}\) |
| `BIAS_CALIB_WEIGHT_LAST_M`, `BIAS_CALIB_WEIGHT_STEP_SESSIONS`, `BIAS_CALIB_WEIGHT_STEP_DROP`, `BIAS_CALIB_WEIGHT_MIN` | Recency-веса |
| `LINEAR_CALIB_B_OFFSET` | Смещение intercept после linear fit |
| `BIAS_CALIB_STD_EPS` | Защита деления при оценке масштаба |

### 8.5 Monotonic reliability

| Имя | Назначение |
|-----|------------|
| `BIAS_MONOTONIC_RELIABILITY_MIN_POINTS` | Минимум точек для включения кривой |
| `BIAS_MONOTONIC_RELIABILITY_N_ZONES` | Число зон (см. fit в коде) |

### 8.6 Квантили для текстовых зон

| Имя | Назначение |
|-----|------------|
| `BIAS_QUANTILE_PERCENTILES` | Перцентили для `q15`…`q85` в JSON модели |
| `_bias_direction_and_strength` | Direction: `≤q45` Bearish, `q45–q55` Neutral/Range, `>q55` Bullish; сила — по вложенным порогам |

### 8.7 Метрики отбора и валидации классов

| Имя | Назначение |
|-----|------------|
| `BIAS_DIRECTIONAL_BEAR_THRESHOLD`, `BIAS_DIRECTIONAL_BULL_THRESHOLD` | Пороги bear/neutral/bull для **selection metrics** и **validation** (сейчас **0.0 / 0.0**: фактически знак `y`) |
| `BUCKET_FACT_BEAR`, `BUCKET_FACT_BULL` | Пороги для других bucket-отчётов по факту (где используются) |

### 8.8 Прочее

| Имя | Назначение |
|-----|------------|
| `BIAS_PREDICT_DEBUG_SIGN_SPLIT` | Флаги отладочного вывода |
| `BIAS_VALIDATION_DEBUG_COMPARE_SIGN_SPLIT` | Второй блок отчёта с колонкой `y_bias_pred_sign_split` |

---

## 9. Калибровки: принцип работы

После **raw** предсказания Ridge (с `StandardScaler` по X и по y) строятся несколько версий score; **одна** из них выбирается как **primary** в зависимости от режима и сохраняется в JSON как основная траектория для этой модели.

### 9.1 `raw`

Без постобработки: `y_primary = y_pred_raw`, клип по [-1, 1] на уровне применения.

### 9.2 `sign_split`

1. **Median alignment** по окну `BIAS_CALIB_K_LAST_N` с recency-весами; опционально trim `y_true` по `BIAS_CALIB_K_TRIM_Q`.
2. Оценка **\(k_{pos}\), \(k_{neg}\)** по обучающим парам (раздельный масштаб положительной и отрицательной стороны), кламп `BIAS_CALIB_K_MIN`…`BIAS_CALIB_K_MAX`.
3. Масштабирование знака: положительные прогнозы × \(k_{pos}\), отрицательные × \(k_{neg}\).
4. **`_bias_apply_neg_asymmetry`** — сжатие хвостов (кусочно-линейная «симметричная» компрессия).
5. **Post shift** с коэффициентом `BIAS_CALIB_POST_SHIFT_FACTOR`.
6. Клип [-1, 1].

Параметры сохраняются в `scale_calibration` JSON (в т.ч. `k_pos`, `k_neg`, median shifts).

### 9.3 `monotonic_reliability`

На **OOF** raw-предсказаниях train (в **`_run_offline_selection_and_save`** фиксировано **`TimeSeriesSplit(n_splits=5)`**; в **`train_bias_model`** **`n_splits=min(TIME_SERIES_SPLITS, max(2, len(y)−1))`**) для пары **`(y_fact, ŷ_raw)`** с теми же **recency-весами**, что median/k-схемы, считаются **пять порогов** по каждой оси методом **`weighted_quantile_5_zone_interpolation`** (середины перцентильных полос \(0\text{–}20,\ldots,80\text{–}100\)). Из пар \((x_i,y_i)\) строится **неубывающая по зонам** ожидаемая кривая за счёт **`np.maximum.accumulate`** по упорядоченным порогам **факта**, затем используется **кусочно-линейная интерполяция** **`raw → E[y|raw]`** с клипом в \([-1,1]\) (`_bias_apply_monotonic_reliability`). **PAVA в коде намеренно не вызывается** (см. докстринг **`_bias_fit_monotonic_reliability`**). Режим сохраняется как **disabled**, если точек мало, недостаточно уникальных \(\hat{y}\), либо после нормализации длина пороговых массивов не совпадает с ожидаемым числом зон (**`BIAS_MONOTONIC_RELIABILITY_N_ZONES`**).

### 9.4 `hybrid`

Смесь **sign_split** и **monotonic** с весами `BIAS_HYBRID_*`. При неработающем monotonic — только sign_split.

### 9.5 `linear`

На последних **`BIAS_CALIB_K_LAST_N`** парах **`(ŷ_raw, y_true)`** с recency-весами **`_bias_recency_weights_for_window`** оцениваются **\(a, b\)** (`_bias_linear_calibration_ab`). При **инференсе** по сохранённому JSON модели: **`a_eff = 1 + 0.6·(a−1)`**, **`b_eff = b + LINEAR_CALIB_B_OFFSET`**, затем \(\mathrm{clip}(a_eff·ŷ_{raw}+b_eff, −1, 1)\) (`_apply_bias_model_to_features`). В процедуре **offline selection** промежуточная калибровка держателя holdout делает клип **`a_eff·ŷ_raw + b` без добавления смещения `LINEAR_CALIB_B_OFFSET`** к intercept — это малое расхождение с финальным production-применителем, которое уже учитывает константу (сравните `_run_offline_selection_and_save` ↔ `_apply_bias_model_to_features`).

### 9.6 Метки primary_bias / bias_strength в ответе модели

По **primary** числовому score и **квантилям**, сохранённым в JSON этой модели (`quantiles`), вызывается `_bias_direction_and_strength`: направление (Bearish / Neutral / Bullish) и сила (Weak/Normal/Strong/Extreme, Range).

---

## 10. Обучение Ridge (`train_bias_model`)

- Вход: `DataFrame` с колонками признаков и целевой **`target_y_column`** (например `y_bias_dir`, `y_bias_strength`, или временный `y_bias` в staging).
- **Пропуски**: `_fill_missing_neighbor_mean` — только **forward-fill** по строкам упорядоченного исторического пайплайна; затем остаточные NaN — **по столбцовым средним train** и `0.0`.
- **Двойное масштабирование**: StandardScaler на X и на y; Ridge на масштабированном y; обратное преобразование предсказания.
- В JSON пишутся: коэффициенты, intercept, скейлеры **X/y**, **`quantiles`** из исторического таргета **этого** запуска, сохранённые наборы параметров для **sign-split**, **monotonic**, **linear** (`a`, `b`, `a_eff`), **hybrid**, выбранный режим **`primary_calibration_mode_config`**, диагностические сводки. При вызове из **`train_predict_bias_segments`** в каждый проход **`train_bias_model`** передаётся **`primary_calibration_mode=_bias_primary_calibration_mode_for_target(kind)`**, поэтому файл **`bias_regression_model_dir_k.json`** несёт калибровочную конфигурацию режима **`BIAS_DIRECTION_*`**, а **`bias_regression_model_strength_k.json`** — **`BIAS_STRENGTH_*`** — это же различают квантильные пороги между цепочками.

---

## 11. Offline feature selection (`_run_offline_selection_and_save`)

Запускается **дважды** при `RUN_OFFLINE_SELECTION=True`:

1. **`target_y_column=y_bias_dir`**, `selection_target_mode="direction"`, калибровка отбора = `BIAS_DIRECTION_PRIMARY_CALIBRATION_MODE` → **`bias_regression_hyperparams_dir.json`**
2. **`target_y_column=y_bias_strength`**, `selection_target_mode="strength"`, калибровка = `BIAS_STRENGTH_PRIMARY_CALIBRATION_MODE` → **`bias_regression_hyperparams_strength.json`**

### 11.1 Разбиение train / test

- Уникальные `predict_date` по возрастанию.
- `n_test = min(N_TEST_SESSIONS, max(0, len(dates) − 2))` (если дат нет — 0; см. `_run_offline_selection_and_save`; верхняя граница гарантирует, что в train остаётся хотя бы возможность непустого ряда при малых выборках).
- Тест — **последние** `n_test` уникальных дат; train — все предыдущие.

### 11.2 Подбор α

`grid_search_bias_hyperparameters`: `GridSearchCV` + `TimeSeriesSplit(TIME_SERIES_SPLITS)`, scoring **`neg_mean_absolute_error`**, сетка `RIDGE_ALPHA_GRID`, опционально усечение числа дат в train до `N_SESSIONS_FOR_HP` / override.

### 11.3 Итеративный leave-one-out pruning

Цикл раннего выхода: если после разбиения **`len(reg_train) < 10`**, **`len(reg_test) < 5`** или в наборе **`len(feat_cols) < 8`**, итерации отбора прекращаются (см. `_run_offline_selection_and_save`). Пока число признаков **> `MIN_FEATURES_AFTER_SELECTION`**:

1. Зафиксировать **α** для текущего набора.
2. Обучить baseline на всех оставшихся признаках; получить raw OOF train + raw test predictions.
3. Применить **ту же primary calibration**, что в production для данного таргета (`_apply_primary_calibration` внутри selection).
4. Снять **holdout metrics** `_bias_test_selection_metrics`.
5. Для каждого кандидата «убрать одну фичу» — повторить; выбрать лучший по rank key.
6. Early stop: если лучшее удаление **заметно хуже** baseline (`_noticeably_worse`: совместное падение Spearman и DirAcc + рост MAE).

### 11.4 Метрики holdout

На **откалиброванных** предсказаниях:

- **`test_balance_penalty`**: \(|pred\_bull\_frac - fact\_bull\_frac| + |pred\_bear\_frac - fact\_bear\_frac|\) при порогах **`BIAS_DIRECTIONAL_*`** (0 → по знаку).
- **`test_spearman`**
- **`test_dir_acc`**: совпадение дискретных классов bear/neutral/bull по тем же порогам.
- **`test_bucket_mono`**: монотонность средних `y_true` по бакетам сортировки по `y_pred`
- **`test_mae`**

Плюс диагностические поля min/median/max предсказаний и доли bear/bull на holdout.

### 11.5 Ранжирование лучшего run

Собирается пул «начальный набор» + состояние после каждого принятого удаления. Лексикографический ключ зависит от `selection_target_mode`:

- **direction**: `(−DirAcc, −Spearman, BalancePenalty, MAE, −BucketMono)`
- **strength**: `(−Spearman, −BucketMono, BalancePenalty, −DirAcc, MAE)`

Выбирается лучший run среди наборов с **числом признаков ≥ `MIN_FEATURES_AFTER_SELECTION`**.

---

## 12. Многостадийное предсказание (`train_predict_bias_segments`)

Функция **`train_predict_bias_segments`** — центральный производственный пайплайн: на выходе восемь JSON стадионных моделей (перезаписываются каждый запуск), аудит, агрегаты для chop и «комбинированные» скоры.

### 12.1 Входные данные и подготовка

1. **`regression_df`**  
   Либо передан снаружи, либо подгружается/достраивается через **`update_bias_regression_table_cache_incremental`**. Если задан **`exclude_predict_session_date`**, строка с этой датой **удаляется** из фрейма перед обучением (`predict_date != exclude`).

2. **Гиперпараметры**  
   Читаются два JSON: **`bias_regression_hyperparams_dir.json`** и **`bias_regression_hyperparams_strength.json`** (поля **`alpha`**, **`selected_features`** / **`feature_order`**). Если список признаков для цепочки пуст или не пересекается с колонками `regression_df`, подставляются **все** колонки из **`REGRESSION_FEATURE_ORDER`**, присутствующие в таблице и **не** входящие в **`FEATURES_EXCLUDED_FROM_MODEL`**.

3. **Одна строка базовых признаков на день прогноза**  
   Вызов **`_bias_prepare_current_features_for_segments(predict_date, data_dir, forecast_session_date=…)`** возвращает **`(current_features, prev_features)`** — словари признаков для **D** и (при наличии данных) для оформления контекста предыдущей сессии, используемого в **`_apply_bias_model_to_features`** как **`features_prev`** (калибровки/некоторые режимы могут опираться на сравнение с прошлым днём). Эта строка **не** берётся как готовая строка из `regression_df`: она **вычисляется заново** из актуальных CSV (см. **`get_bias_regression_features_for_date`** внутри подготовки). В **базовые** ключи **не входят** столбцы сегментных `bias_dir_k` / `bias_strength_k` — они появляются только как **дополнительные** признаки начиная со стадии 2 (§6.3).

### 12.2 Порядок циклов обучения и инференса

В коде два вложенных цикла:

```text
for stage in (1, 2, 3, 4):
    for target_kind in ("dir", "strength"):
        # обучение + инференс для этой пары (stage, target_kind); см. перечень шагов ниже.
```

То есть для **каждой** стадии по очереди обрабатывается **полностью параллельная пара** modality: сначала **direction** данной стадии (обучение + предсказание на `predict_date`), затем **strength** данной стадии. Это сохраняет независимость двух шкал: **модель strength не использует выход модели dir** ни на одном этапе — только **`bias_strength_*`** своей цепочки в extra-признаках. Режим **`primary_calibration_mode`**, передаваемый одновременно в **`train_bias_model`** и **`_apply_bias_model_to_features`**, выбирается **по цепочке**: **`_bias_primary_calibration_mode_for_target("dir")`** и **`...("strength")`**.

На каждом шаге (`stage`, `target_kind`):

1. Строится обучающий датафрейм **`stage_df` = `_bias_stage_training_view(regression_df, stage, kind)`**, внутри него же вызывается **`_bias_segment_table_from_base`**: к базовым фичам таблицы добавляются столбцы сегментов и вычисляемые **mean** для стадий 3–4 так, как в §6.3 (**на исторических строках всё считается от фактов** колонок `bias_*`).
2. Формируется список колонок X: **`feature_cols = _bias_stage_feature_cols(selected_k, stage, stage_df, kind)`** — сохранённые из hyperparams признаки **плюс** одна optional extra-колонка стадии, если объявлена в **`_bias_stage_extra_feature`**; колонки из **`FEATURES_EXCLUDED_FROM_MODEL`** по-прежнему исключаются.
3. Вызывается **`train_bias_model(..., target_y_column="y_bias", model_path=bias_regression_model_{kind}_{stage}.json, primary_calibration_mode=calibration_mode, ...)`**, что **перезаписывает** JSON этой стадии (Ridge payload + калибровки + метаданные). Здесь **`calibration_mode`** — значение из **`_bias_primary_calibration_mode_for_target(kind)`**.
4. Строится копия словаря признаков для инференса: **`features_for_stage = dict(current_features)`**.
5. В **`features_for_stage`** подставляются значения из **`stage_outputs`** по правилам §6.3 (для нового дня — **откалиброванный** **`y_bias`** предыдущих стадий **того же kind**).
6. Вызов **`_apply_bias_model_to_features(features_for_stage, model_path, features_prev=prev_features, primary_calibration_mode=calibration_mode)`**: по порядку ключей **`feature_order`** из JSON для каждого признака берётся значение из словаря; при **NaN** сначала подставляется одноимённый ключ из **`features_prev`** (предыдущая сессия), иначе — **среднее этого признака из обучающего StandardScaler** модели. Строятся **все** траектории (raw, sign_split, monotonic, hybrid, linear), а **выходной** **`y_bias`** — это **primary** по переданному режиму или по полю модели (см. ветвление в коде). Результат кладётся в **`stage_outputs[kind][stage]`**.

После того как все 4×2 шага выполнены, из **`stage_outputs`** извлекаются числовые \(\hat{b}\) для **`bias_dir_1`–`bias_dir_4`** и **`bias_strength_1`–`bias_strength_4`** (везде значение ключа **`y_bias`** внутри вложенных словарей = **primary**).

### 12.3 Взвешенная агрегация по сегментам для chop (**`bias_dir_for_chop`**, **`bias_strength_for_chop`**)

Отдельно для цепочек **direction** и **strength** применяется **одна и та же** схема взвешивания по длительности сегментов (**`bias_weighted_segments_four`**):

\[
V = \frac{w_1 v_1 + w_2 v_2 + w_3 v_3 + w_4 v_4}{w_1+w_2+w_3+w_4}
      = \frac{90\,v_1 + 90\,v_2 + 90\,v_3 + 120\,v_4}{390},
\]

где \(v_k = \hat{b}_k\) — **primary после калибровки** (поле **`y_bias`** после стадии \(k\) в цепочке **dir**, либо **strength**). Константы \(w_k\) — **`BIAS_SEGMENT_*_MINUTES`**, знаменатель **`RTH_MINUTES_PER_SESSION`**.

Так задаются два «тонких» входа chop/TradePlan-only: **наклонность** и **энергичность движения** в родных шкалах Ridge, **без** сведения dir+str в общую решётку `bias_raw_to_unified_score`.

### 12.4 «Комбинированная» шкала: единые дискретные уровни + вес и **`bias_for_chop`**

Пайплайн вызывает **`bias_prediction_combined_audit(regression_df, bd1, …, bd4, bs1, …, bs4)`**, куда \(\hat{b}\) подставлены уже **из primary** (**не** из **`y_bias_raw`**). Последовательность в коде такая:

1. **`bias_build_combined_hist_quantiles_pack(regression_df)`** (см. следующий блок) всегда вызывается **первой** строкой аудита и кладётся в ключ **`bias_combined_hist_quantiles`** результата.
2. Для каждого \(k\) из набора \(\{1,2,3,4\}\) читаются **квантильные словари** именно **`bias_regression_model_dir_k.json`** и **`bias_regression_model_strength_k.json`** через **`bias_stage_model_quantiles`**. Это пороговые уровни **обучающего таргета** соответствующей стадии (те же ключи **`q15`…`q85`**, что пишутся в JSON при **`train_bias_model`**).
3. **`bias_raw_to_unified_score(y, quantiles)`** переводит действительное \(\hat{y}\) (вход сюда уже **primary** стадионной модели, а не её «сырой» dot-product) в **ступень** \(\{-1, -0.75, \ldots, +1\}\) по порогам **`q15`…`q85`**: \(\le q15 \to -1\), \(\le q25 \to -0.75\), \(\ldots\), \(> q85 \to +1\) (точные границы — в коде функции).
4. **`bias_segment_combined_from_raw(rd, rs, q_dir, q_str)`** = **`BIAS_COMBINED_DIR_WEIGHT · unified(dir) + BIAS_COMBINED_STRENGTH_WEIGHT · unified(strength)`** (константы **`BIAS_COMBINED_DIR_WEIGHT`**, **`BIAS_COMBINED_STRENGTH_WEIGHT`** в `bias.py` — см. код; условно можно читать как **перекос комбинации к каналу strength**). Если unified для одного канала недоступен (нет квантилей или нечисловой вход), весь **`bias_combined_k`** становится **NaN**.
5. **`bias_for_chop`** = **`bias_segment_combined_from_raw(`** **`bias_dir_for_chop`**, **`bias_strength_for_chop`**, **`q_dir` из стадии 4**, **`q_strength` из стадии 4** **`)`** — та же цепочка unify + веса **`BIAS_COMBINED_*`**, что и для каждого **`bias_combined_k`**, но один раз на уже усреднённые по минутам chop-скаляры.

Так **`bias_for_chop`** **алгебраически согласован** с отображёнными в аудите **`bias_dir_for_chop`** и **`bias_strength_for_chop`** (после решётки unify они дают один combined-скор). Посегментные **`bias_combined_k`** остаются отдельными показателями по окнам \(k\).

#### 12.4.1 Эмпирический пакет **`bias_build_combined_hist_quantiles_pack`**

По **всем строкам переданного** `regression_df` для каждого сегмента \(k\) и для финального **`chop_combined`**:

- Берётся строка истории (**фактический** таргет в таблице) **`bias_dir_k`**, **`bias_strength_k`**.
- С тем же картографированием, что для прогноза сейчас, считаются скоры **`bias_segment_combined_from_raw`**; по накопленным выборкам строится словарь эмпирических квантилей через **`bias_historical_quantiles_dict_from_scores`** (**тот же набор процентней `BIAS_COMBINED_HISTORY_QUANTILE_PROBS`** и ключи **`BIAS_COMBINED_HISTORY_QUANTILE_KEYS`** = `q15`…`q85`), если накопилось \(\ge 8\) конечных точек для данного множества; иначе словарь **пустой**.
- Ключ **`"chop"`** в пакете — распределение **`bias_for_chop`** по историческим строкам: для каждой строки минутное усреднение **`bias_dir_1..4`** и **`bias_strength_1..4`**, затем **`bias_segment_combined_from_raw`** с квантилями стадии 4 (как в прогнозе).

**Важно:** квантили здесь считаются **после** того, как на диск уже лежат актуальные JSON стадии текущего прогона: при серийном переобучении всех **`bias_regression_model_*_k.json`** упакованные пороги согласуются с **новыми** `quantiles` внутри этих файлов. Табличные факты **`bias_*_k`** остаются **историческими** (это записанные истинные сегменты прошлых дней).

### 12.5 Аудит и консоль

При **`save_audit=True`** сохраняется **`bias_last_prediction_debug.json`** (если файл не указан явно иначе) с полным подмножеством результата: стадионные \(\hat{b}\), **`bias_*_for_chop`**, **`bias_combined_*`**, дерево **`bias_combined_hist_quantiles`**. Точный список полей см. сборку **`audit_payload`** внутри **`train_predict_bias_segments`**.

**`format_bias_segments_console`** (вывод из **`main`**) для каждой стадии печатает числа из верхнего уровня результата и текстовые зоны **dir/strength** из вложенных **`stage_predictions`**, а для **combined** и для **bias_for_chop** берёт исторические словари квантилей **`bias_combined_hist_quantiles[str(k)]`** и **`["chop"]`**, после чего прогоняет **`_bias_direction_and_strength`** — человек видит \(\hat{y}\) не против квантилей одной стадии ridge, а против **эмпирических** порогов historical combined тем же методом зонирования.

---

## 13. Публичный API `predict_bias_and_probability`

- Вызывает **`train_predict_bias_segments`** (все эффекты обучения/перезаписи стадионных JSON и аудита — внутри него).
- Возвращает словарь для TradePlan / chop / отчётов. Ключевые моменты:
  - **`y_bias`** = **`bias_for_chop`** (combined на **`bias_dir_for_chop`** / **`bias_strength_for_chop`**, квантили стадии 4, см. §12.4);
  - **`bias_dir_for_chop`**, **`bias_strength_for_chop`** — минутное взвешивание **четырёх откалиброванных** сегментных primary (§12.3);
  - **`bias_combined_1`–`bias_combined_4`** и пакет **`bias_combined_hist_quantiles`** — из **`bias_prediction_combined_audit`**;
  - числовые **`bias_dir_1`–`bias_dir_4`**, **`bias_strength_1`–`bias_strength_4`** — проброшены из **`train_predict_bias_segments`** теми же ключами (**primary** каждой стадии); также **`forecast_session_date`** в ответе, если параметр задавался;
  - текстовые **`primary_bias`** и **`bias_strength`** — из словаря предсказания **стадии 4 direction** (`stage_predictions["bias_dir_4"]`), интерпретация **дискретных зон именно финального сегмента direction** Ridge; топ-уровневый блок **`quantile_range`/`quantiles`** синхронизирован с тем же объектом (**не** со шкалой **`bias_for_chop`** из historical combined).
- Аргументы **`model_path`**, **`primary_calibration_mode`** в сигнатуре оставлены для совместимости старых вызовов; в текущей логике они **игнорируются** (см. докстринг в `bias.py`: отдельная pooled-модель под эти аргументы в стадийном режиме **не** подменяет пайплайн).

---

## 14. Режимы `if __name__ == "__main__"`

### 14.1 Daily (`RUN_OFFLINE_SELECTION=False`)

- Требуются **оба** валидных файла `bias_regression_hyperparams_dir.json` и `bias_regression_hyperparams_strength.json`.
- Инкремент / обновление кэша таблицы (с учётом **`exclude_predict_session_date`**, если передан **`--plan-for`** / `BIAS_CLI_SESSION`).
- Перед стадийным циклом выполняется переобучение **двух pooled** моделей на **полных** таргетах **`y_bias_dir`** и **`y_bias_strength`** на подмножестве строк **без** excluded-даты (если она задана) — результаты в **`bias_regression_model.json`** и **`bias_regression_model_strength.json`**. Эти pooled JSON используются в daily/диагностических ветках; **стадийный** прогноз для Trade Plan идёт через **`train_predict_bias_segments`**.
- Затем **`train_predict_bias_segments`** (в консольный лог — в т.ч. **`format_bias_segments_console`**, аудит).

### 14.2 Offline (`RUN_OFFLINE_SELECTION=True`)

- Строка в лог: **`Building unified bias regression table (features + session/stage targets)...`**
- Полная пересборка таблицы → при непустом результате в консоль выводится **`Regression table shape: (n_rows, n_cols) (train_dates=…, test_dates=…)`**, где **train_dates** / **test_dates** — число **уникальных** `predict_date` в train и holdout при том же правиле, что **`_run_offline_selection_and_save`** (`OFFLINE_N_TEST_SESSIONS` или **`N_TEST_SESSIONS`**, с отсечением `max(0, len(dates)−2)` для теста); затем **`Regression table saved: bias_regression_table.csv.gz`**.
- Далее — сводки пропусков и корреляции признаков, два прогона **offline selection**.
- Финальные **`train_bias_model`** на полный `reg_df` для pooled **`bias_regression_model.json`** / **`bias_regression_model_strength.json`**.
- **`train_predict_bias_segments`** для дня из **`get_predict_date_from_data()`** (или с учётом CLI-сессии / `forecast_session_date`, см. блок `main`).

---

## 15. Валидация и отчёты

- **`evaluate_model_buckets`**: сравнение `y_bias_pred` и `y_bias_fact` (пороги направления по умолчанию 0), Spearman, direction accuracy, MAE, построение **бакетов** по предсказанию; опционально второй блок для sign-split при `BIAS_VALIDATION_DEBUG_COMPARE_SIGN_SPLIT`.
- **`_bias_stage_validation_report`**: для каждой стадии/kind строится holdout-таблица через `build_historical_validation_table_from_regression_df` и печатаются buckets.
- **`print_bias_validation_summary`**: краткий отчёт без бакетов (удобно для daily).

---

## 16. Указатель ключевых функций

| Область | Функции |
|---------|---------|
| Таблица | `build_bias_regression_table`, `update_bias_regression_table_cache_incremental`, `_build_bias_regression_rows_for_dates` |
| Признаки на D | `get_bias_regression_features`, `get_bias_regression_features_for_date` |
| Таргеты сессии | `_y_bias_direction_strength_session`, `_bias_segment_targets_from_block`, `_daily_atr14_for_date` |
| Staging | `_bias_segment_table_from_base`, `_bias_stage_training_view`, `_bias_stage_model_path` |
| Пропуски | `_fill_missing_neighbor_mean` |
| HP search | `grid_search_bias_hyperparameters` |
| Обучение | `train_bias_model` |
| Калибровки | `_bias_apply_*`, `_bias_fit_monotonic_reliability`, `_bias_sign_split_calibration_ks`, `_bias_primary_calibration_mode_for_target` |
| Унификация для combined | `bias_raw_to_unified_score`, `_bias_quantiles_from_model` |
| Отбор | `_run_offline_selection_and_save`, `_bias_test_selection_metrics`, `_pick_best_bias_run_index` |
| Инференс | `_apply_bias_model_to_features`, `train_predict_bias_segments`, `predict_bias_and_probability` |
| Комбинированные скоры и chop-агрегаты | `bias_stage_model_quantiles`, `bias_segment_combined_from_raw`, `bias_prediction_combined_audit`, `bias_weighted_segments_four`, `bias_build_combined_hist_quantiles_pack`, `bias_historical_quantiles_dict_from_scores` |
| Вывод оператору | `format_bias_console`, `format_bias_segments_console` |
| Валидация | `build_historical_validation_table_from_regression_df`, `evaluate_model_buckets`, `print_bias_validation_summary` |

---

## 17. `bias_test_models.py` (кратко)

Рядом с основным пайплайном лежит модуль **[`bias_test_models.py`](../bias_test_models.py)** — **вспомогательный контур для экспериментов и отчётности**, а не часть ежедневного `predict_bias_and_probability`. По сути это «песочница» для сравнения вариантов модели (разные калибровки/конфигурации), построения исторических таблиц валидации (**`build_historical_validation_table`**, **`evaluate_model_buckets`** и связанные отчёты) и проверки качества на отложенной выборке без смешения с боевым стадийным прогнозом TradePlan. Идея зафиксирована здесь, чтобы не путать файл с производственным `bias.py`.

---

*Конец документа. Обновляйте этот файл при изменении констант, таргетов, схемы стадий или процедуры отбора в `bias.py`.*

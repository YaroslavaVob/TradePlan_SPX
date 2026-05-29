# Алгоритм расчёта уровней поддержки и сопротивления (модуль `levels.py`)

Документ описывает **текущую** реализацию поиска levels в `levels.py` (flat-segment pipeline). Ниже также сохранён справочный раздел про legacy L1/L2 + DBSCAN pipeline.

Ключевая идея: уровни строятся из **ATR-нормированного detecтора flat-сегментов** (15m) по историческим сессиям, превращаются в **level-candidates** (center + band), затем **группируются** в grouped levels и проходят **вторичный пост-merge**.

## 0. Недавние улучшения (важно)

В `levels.py` добавлены улучшения, которые влияют на качество финальных уровней и их ранжирование:

- **Regime-aware merge tolerances**: `merge_center_tol` для группировки кандидатов зависит от `median_width`, `mean_range_spx` и `median_atr_ref`.
- **Shelf detector (group-level reaction stats)**: на grouped-level считаются агрегаты реакций по сессиям (`session_touch_count`, `bounce_count`, `break_count`, `flip_count`, `reaction_flip_count`, `mean_excursion_after_touch_*`) и whipsaw-marker `two_sided_excursion_count`.
- **Final suppression step (min spacing)**: после post-merge применяется подавление слишком близких финальных уровней по regime‑зависимой минимальной дистанции.
- **Overcrowding / proximity penalty**: финальный скоринг включает `distance_penalty` за overcrowding и штраф по `two_sided_excursion_count`.

## 1. Входные данные

Используются:
- `open_pred` — прогноз открытия/ориентир на уровне.
- `mean_range_spx` — средний диапазон SPX (в пунктах).
- `spx_1m_rth` — минутные OHLC по SPX только для RTH (09:30–15:59 NY), историческая база.
- `es_1m` — минутные данные ES (нужны для overnight 00:00–09:30 NY).

Параметры времени:
- Для **SPX** анализируются сессии по `Date` (вход уже RTH-срез).
- Для **ES** анализируется overnight на каждой `session_date`: `00:00 <= t < 09:30 NY`.

## 2. Диапазон поиска кандидатов вокруг `open_pred`

Используется полоса:
- `pred_lo = open_pred - LEVEL_RANGE_MULT * mean_range_spx`
- `pred_hi = open_pred + LEVEL_RANGE_MULT * mean_range_spx`

Константа:
- `LEVEL_RANGE_MULT = 1.25`

Кандидат уровня принимается, если выполняется одно из условий:
- `level_price (center) ∈ [pred_lo, pred_hi]`
- или **band** кандидата (через `z_lo/z_hi`) пересекает `[pred_lo, pred_hi]` с `overlap_ratio >= 0.50`, где:
  `overlap_ratio = overlap_length / band_width`, `overlap_length` — длина пересечения по цене, `band_width = z_hi - z_lo`.

## 3. Flat-window detector (15 минут) по ATR

### 3.1 Окно 15m

Для каждой исторической сессии (SPX) и для каждой ES overnight-секции выполняется сканирование rolling windows длиной:
- `LEVEL_FLAT_WINDOW_MINUTES = 15`

Для каждого окна:
- `window_range = max(High_w) - min(Low_w)`
- `atr_w = median(ATR Wilder(14) внутри окна)`
- `drift = abs(close_last - close_first)`
- `close_spread = q90(close_w) - q10(close_w)`

Окно считается **flat**, если одновременно:
- `window_range <= LEVEL_K1_FLAT_ATR * atr_w` , `LEVEL_K1_FLAT_ATR = 5.0`
- `drift <= LEVEL_K2_TREND_BLOCK * window_range` , `LEVEL_K2_TREND_BLOCK = 0.35`
- `close_spread <= LEVEL_K3_PRICE_DENSITY * atr_w` , `LEVEL_K3_PRICE_DENSITY = 3.5`

ATR:
- Wilder ATR(14) считается по барам 1m с той же инициализацией, что в `chop.py`:
  первый ATR появляется после прогрева (см. `LEVEL_ATR_INIT_BARS = 15`).

### 3.2 Склейка окон внутри одной сессии

После нахождения flat-окон их интервалы объединяются внутри сессии:
- `LEVEL_FLAT_MERGE_GAP_MINUTES = 3`

То есть объединяется соседнее окно, если оно **пересекается** либо разрыв между ними не больше 3 минут.

Результат: набор **merged flat-segments**.

## 4. Превращение segment в level-candidate

Для каждого merged segment считаются:
- `z_lo = q10(close in segment)`
- `z_hi = q90(close in segment)`
- `center = median(close in segment)`
- `width = z_hi - z_lo`
- `duration_min = длина segment в минутах`
- `inside_share = доля значений close в [z_lo, z_hi]`
- `atr_ref = median(ATR внутри segment)`

Дополнительные признаки сегмента:
- `touch_count`: число баров, где `[Low_i, High_i]` пересекает `center` (Low <= center <= High)
- `touch_density = touch_count / duration_min`
- `rejection_score`:
  `max_i |Close_post_i - center| / atr_ref`,
  где рассматривается горизонт после окончания сегмента `LEVEL_N_AFTER_MIN = 10` минут.

### 4.1 Геометрия уровня (band) для наружного API

Финальным объектом для `levels.py` на уровне candidate является геометрия:
- `level_price = center`
- `level_band_halfwidth = max(width/2, LEVEL_BAND_C1 * atr_ref)`
- `band_lo = center - level_band_halfwidth`
- `band_hi = center + level_band_halfwidth`

Константа:
- `LEVEL_BAND_C1 = 0.50`

### 4.2 Фильтр качества сегмента

Сегмент сохраняется, если выполняется **одно** из двух условий:

**A. Acceptance-segment**
- `duration_min >= 15`
- `inside_share >= 0.80`

**B. Dense short level-segment**
- `duration_min >= 10`
- `width <= LEVEL_MAX_WIDTH_MULT_ATR * atr_ref` , `LEVEL_MAX_WIDTH_MULT_ATR = 1.8`
- `touch_count >= touch_min`, где:
  `touch_min = max(3, ceil(duration_min * 0.4))`

### 4.3 Post-touch reaction features (для shelf-агрегации)

Для дальнейшей агрегации на уровне price shelf (group-level) на сегменте сохраняются признаки реакции в пост-окне `LEVEL_N_AFTER_MIN` (в ATR-единицах):

- `reaction_max_up_atr`: максимальное \( (Close_{post} - center) / atr_ref \)
- `reaction_max_down_atr`: минимальное \( (Close_{post} - center) / atr_ref \)
- `reaction_max_abs_atr`: максимальное абсолютное отклонение (используется как `rejection_score`)
- `reaction_end_diff_atr`: \( (Close_{post,last} - center) / atr_ref \)
- `reaction_end_dir`: знак `reaction_end_diff_atr` (−1/0/+1)
- `reaction_crossed_center`: был ли переход через центр в post-window (diff менял знак / касался обеих сторон)

## 5. ES candidates и basis (ES → SPX эквивалент)

ES overnight-сегменты извлекаются из `es_1m` для каждой `session_date` в окне `00:00–09:30 NY`.

Перед применением фильтра вокруг `open_pred` ES цены переводятся в SPX-эквивалент:
- `basis_d = Close_last_ES(prev_session) - Close_last_SPX(prev_session)`
- `spx_price = es_price - basis_d`

где `prev_session` выбирается по SPX calendar (предыдущая SPX RTH сессия относительно текущей `session_date`).

Далее:
- к translated ES candidates применяется тот же фильтр вокруг `pred_lo/pred_hi`.

## 6. Группировка historical candidates в grouped levels

Кандидаты объединяются (в одном общем SPX-пространстве) в grouped levels.

### 6.1 Критерии объединения

Для группировки кандидатов используется адаптивный `merge_center_tol` (regime-aware):

- `merge_center_tol = max(`
  - `LEVEL_MERGE_CENTER_BASE_TOL`,
  - `0.20 * median_width`,
  - `0.06 * mean_range_spx`,
  - `0.8 * median_atr_ref`
  `)`

Константа:
- `LEVEL_MERGE_CENTER_BASE_TOL = 1.5`

И требование по перекрытию band:
- `overlap_min = LEVEL_MERGE_OVERLAP_MIN = 0.35`
- `overlap_ratio = overlap_length / min(width_i, width_j)`

Если для пары кандидатов выполнены оба условия:
- `abs(center_i - center_j) <= merge_center_tol`
- `overlap_ratio >= overlap_min`

то они попадают в одну компоненту объединения (union-find).

### 6.2 Центр и ширина grouped-level

Для grouped-level:
- `width_final = median(all band_width in group)`
- `center_final` считается как **weighted average** центров кандидатов:
  `weights_i = inside_share_i * duration_min_i * max(touch_density_i, 1e-6)`
  `center_final = average(center_i, weights=weights_i)`

После этого:
- `band_lo = center_final - width_final/2`
- `band_hi = center_final + width_final/2`

### 6.3 Частоты по источникам (SPX / ES) и weighted rank

Для группы отдельно считаются:
- `spx_session_count = число unique session_date среди candidate.source_type=="SPX"`
- `es_session_count = число unique session_date среди candidate.source_type=="ES"`
- `spx_segment_count = количество segment/candidate source_type=="SPX"`
- `es_segment_count = количество segment/candidate source_type=="ES"`

Weighted-частоты:
- `weighted_session_frequency = SPX_SOURCE_WEIGHT * spx_session_count + ES_SOURCE_WEIGHT * es_session_count`
- `weighted_segment_frequency = SPX_SOURCE_WEIGHT * spx_segment_count + ES_SOURCE_WEIGHT * es_segment_count`

Константы:
- `SPX_SOURCE_WEIGHT = 1.35`
- `ES_SOURCE_WEIGHT = 1.00`

Сила уровня (strength) на grouped-level:
- `strength = mean_inside_share * mean_touch_density * (1 + max(0, mean_rejection_term))`

Сторона:
- `side = support`, если `center_final < open_pred`
- `side = resistance`, если `center_final > open_pred`

Ранжирование grouped-level:
1) `weighted_session_frequency` desc
2) `weighted_segment_frequency` desc
3) `mean_inside_share` desc
4) `median_width` asc (в коде это `width_final`)

### 6.4 Shelf / price-shelf статистики (group-level reaction stats)

На уровне grouped-level дополнительно считаются статистики реакций по **уникальным сессиям**:

- `session_touch_count`: число сессий, где зона была представлена сегментами (и есть post-reaction)
- `bounce_count`: число сессий с `reaction_max_abs_atr >= LEVEL_SHELF_BOUNCE_MIN_EXCURSION_ATR`
- `break_count`: число сессий с эвристикой “one-sided break” (сильный односторонний уход и нет meaningful cross-back через центр)
- `two_sided_excursion_count`: число сессий с выраженной двусторонней экскурсией (whipsaw-marker)
- `reaction_flip_count`: смены знака реакции (`reaction_end_dir`) между сессиями (directional-only)
- `flip_count`: role-aware flip (support-like vs resistance-like) — смена “ожидаемого” поведения зоны между сессиями
- `mean_excursion_after_touch_atr`: средняя по сессиям `reaction_max_abs_atr`
- `mean_excursion_after_touch_pts = mean_excursion_after_touch_atr * median_atr_ref`

## 7. Second-stage merge (post-merge grouped levels)

Чтобы не “перерассекать” близкие уровни на стадии union-find, выполняется второй этап merge для **уже готовых** grouped levels на одной стороне.

Параметры:
- `POST_LEVEL_MERGE_TOL = 5.0`
- `POST_LEVEL_MERGE_OVERLAP_MIN = 0.15`

Условия объединения соседних уровней (внутри support отдельно и resistance отдельно):
- `abs(price_i - price_j) <= POST_LEVEL_MERGE_TOL`
- `overlap_ratio(band) >= POST_LEVEL_MERGE_OVERLAP_MIN`

Цена merged-level:
- `merge_price = weighted_average(price, weights=weighted_session_frequency)`

Ширина:
- `width_final = median(widths)`

После merge ранги пересчитываются по той же weighted-логике.

Важно: post-merge **не теряет** shelf-метрики — они переносятся/агрегируются в merged output:
- counts (`session_touch_count`, `bounce_count`, `break_count`, `two_sided_excursion_count`, `flip_count`, `reaction_flip_count`) суммируются,
- `mean_excursion_after_touch_atr` агрегируется как среднее с весами по `session_touch_count`,
- `mean_excursion_after_touch_pts` пересчитывается через `median_atr_ref`.

## 7.1 Final suppression step (минимальная дистанция между финальными уровнями)

После post-merge выполняется финальный шаг подавления слишком близких уровней (support отдельно, resistance отдельно).

Минимальная дистанция (regime-dependent):

- `min_spacing = max(`
  - `LEVEL_FINAL_MIN_SPACING_PTS_FLOOR`,
  - `LEVEL_FINAL_MIN_SPACING_RANGE_FRAC * mean_range_spx`,
  - `LEVEL_FINAL_MIN_SPACING_ATR_MULT * median_atr_ref_final`
  `)`

Алгоритм:
- сортируем уровни “сильные→слабые” по `pre_score`,
- если новый уровень ближе `min_spacing` к уже принятому:
  - если overlap band достаточен — **merge**,
  - иначе — **drop** слабый.

## 7.2 Overcrowding penalty и итоговый скоринг

Для финального ранжирования уровню присваиваются поля:

- `pre_score`: используется для initial ordering в suppression
  - `pre_score = strength + c1*session_touch_count + c2*bounce_count + c3*break_count + c4*flip_count + c5*mean_excursion_after_touch_atr`
- `raw_score`: финальная “полезность” до penalty (учитывает также `reaction_flip_count`, `break_count`, `mean_excursion_after_touch_atr` и штрафует `two_sided_excursion_count`)
- `distance_penalty`: penalty < 1, если уровень близко к уже принятым более сильным уровням
- `final_score = raw_score * distance_penalty`

## 8. ES overnight High/Low attach как отдельные уровни

После получения `levels_ranked`, модуль вычисляет ES overnight high/low для `plan_for`:
- `es_low_lvl`, `es_high_lvl` (окно `00:00–<09:30 NY`)

Далее уровни добавляются как **отдельные point-levels** (band width = 0) при отсутствии near-match:
- `ES_HILO_ATTACH_TOL = 0.25`

Логика:
- если `es_low_lvl < open_pred` и рядом с support нет уровня `abs(price - es_low_lvl) <= ES_HILO_ATTACH_TOL`, то добавляется level:
  - `side="support"`, `source_type="ES_HILO"`, `rank="Low ES"`
- если `es_high_lvl > open_pred` аналогично для resistance:
  - `side="resistance"`, `source_type="ES_HILO"`, `rank="High ES"`

Такие уровни:
- попадают в `support/resistance` списки,
- и сохраняются в `levels_meta`/`levels_log["levels_ranked_meta"]`.

## 9. Вывод наружу

Функция **`predict_key_levels`** при заданных конечных **`open_pred`** и **`mean_range_spx`** вызывает **`_levels_from_flat_segments`** (основной путь §1–9). Если переданы **`level_search_lo` / `level_search_hi`** (типично из `level_search_band_from_scenario_targets` по таргетам chop) и после фильтрации не осталось ни одного уровня, выполняется **повторный** вызов с отключённым узким поиском (`pred_search_lo/hi = None`), чтобы не вернуть пустой результат только из‑за «узкого» union-сценариев.

`predict_key_levels(...)` возвращает:
- `support: list[float]` — цены уровней поддержки
- `resistance: list[float]` — цены уровней сопротивления

В логах (для отладки/визуализации) дополнительно сохраняются:
- `levels_log["levels_ranked_meta"]` — список grouped уровня с полями:
  - `price`, `side`, `band_lo`, `band_hi`, `width`, `strength`, `rank`
  - `spx_session_count`, `es_session_count`, `spx_segment_count`, `es_segment_count`
  - `weighted_session_frequency`, `weighted_segment_frequency`
  - `mean_inside_share`, `median_duration`, `median_atr_ref`, `source_type`
  - shelf stats:
    - `session_touch_count`, `bounce_count`, `break_count`, `two_sided_excursion_count`
    - `flip_count`, `reaction_flip_count`
    - `mean_excursion_after_touch_atr`, `mean_excursion_after_touch_pts`
  - scoring (для suppression / final ranking):
    - `pre_score`, `raw_score`, `distance_penalty`, `final_score`

---

## Appendix A (legacy): DBSCAN pipeline (A–F)

Ниже приведён **legacy** пайплайн поиска уровней по этапам **A→F** (L1/L2 + DBSCAN): соответствующий код в `levels.py` сохранён для совместимости и экспериментов. **Текущая** интеграция с `build_TradePlan.py` опирается на **flat-сегментный** детектор из разделов **1–9** (`predict_key_levels` → `_levels_from_flat_segments`).

Отдельно: при очень большом числе центров-кандидатов в flat-пайплайне может включаться **DBSCAN как fallback группировки центров** (порог **`LEVEL_PAIRWISE_MAX_CANDIDATES`**, наличие `sklearn`), после чего всё равно выполняется попарная агрегация уровней — см. исходники `levels.py`.

# Алгоритм расчёта уровней поддержки и сопротивления — DBSCAN (legacy ветка)

Описание **альтернативного** алгоритма по этапам **A→B→C→D→E→F**: сырые зоны L1–L2 (**A**), **дедупликация зон внутри сессии** (**B**), кластеризация **DBSCAN с адаптивными eps и min_samples** (**C**), один уровень на кластер с перцентилями и фильтром (**D**), **объединение близких итоговых уровней** по одной стороне от open_pred и взвешиванием по размеру кластера (**E**), затем **добавление двух ES overnight уровней (High/Low) как дополнительных финальных уровней** (**F**). Источник данных: SPX 1m RTH (`SPX_1min_90d_ibkr.csv`).

---

## 1. Входные данные и диапазон поиска

### 1.1 Источники параметров

- **open_pred** — прогноз открытия сессии (из `open_price.py` или `open_prediction_{date}.json`).
- **mean_range_spx** — средний диапазон сессии в пунктах (из `chop_prediction_{date}.json`, поле `"mean_range_spx"`).
- **spx_1m_rth** — минутные данные SPX только по RTH (регулярная торговая сессия), все доступные исторические сессии (например 90 дней).

### 1.2 Диапазон поиска зон

Задаётся полоса цен, в которой ищем «узкие» зоны только внутри неё:

- **level_range_spx**:  
  `[open_pred - half, open_pred + half]`,  
  где `half = mean_range_spx * LEVEL_RANGE_MULT`.

Константа: **LEVEL_RANGE_MULT = 1.25**.

Пример: `open_pred = 6951`, `mean_range_spx = 79.6` →  
`half = 99.5` → поиск зон в диапазоне **6851.5 – 7050.5**.

---

## 2. Узкие зоны (L1–L2)

По каждой **исторической** сессии (день) в `spx_1m_rth` сканируются минутные бары. Зона считается найденной, если в некотором **скользящем окне** выполнены условия по ширине диапазона (или по телу свечей) и доле баров. Используются две зоны, чтобы избежать дублирования одних и тех же уровней.

Для каждой зоны хранится пара **(zmin, zmax)** — минимум и максимум цены в окне. Перед расчётом `zone_mean` границы зоны **урезаются по квантилям внутри окна**: берём \(q10\) и \(q90\) по `Close` в окне и заменяем границы на:

- `zmin = max(zmin, q10)`
- `zmax = min(zmax, q90)`

Далее используется **zone_mean = (zmin + zmax) / 2** и **zone_width = zmax − zmin** уже по “trim”‑границам.

### 2.1 Критерии зон

| Критерий | Окно (мин) | Ширина / тело | Условие |
|----------|------------|----------------|---------|
| **L1**   | 15         | ≤ 10 п.п.      | ≥ 90% баров: Close ∈ [zmin, zmax] (диапазон окна) |
| **L2**   | 15         | —              | ≥ 80% баров: тело свечи ≤ 2.5 п.п.; зона = [min Low, max High] окна |

Константы в коде:

- L1: `L1_WINDOW=15`, `L1_WIDTH_PTS=10.0`, `L1_PCT=0.90`
- L2: `L2_WINDOW=15`, `L2_BODY_MAX_PTS=2.5`, `L2_PCT=0.80`

### 2.2 Отбор зон по диапазону

По одной сессии (день) собираются зоны L1 и L2. В итоговый список попадают только зоны, **пересекающиеся** с `[range_lo, range_hi]`:

- условие: `zhi >= range_lo` и `zlo <= range_hi`.

Таким образом, зоны за пределами `level_range_spx` отбрасываются.

### 2.3 Представление зон и сортировка

Для каждой зоны хранится:

- **zone_mean** = (zmin + zmax) / 2
- **zone_width** = zmax − zmin
- **source_type** — тип зоны: `"L1"` или `"L2"`
- **session_date** — дата сессии

Зоны сортируются по **zone_mean** (внутри обработки — по паре (session_date, zone_mean)).

### 2.4 Дедупликация зон внутри сессии (до DBSCAN)

Чтобы одна и та же консолидация не учитывалась многократно (десятки почти совпадающих зон из одного дня), перед кластеризацией выполняется **объединение зон в рамках одной сессии**.

**Допуск слияния** (адаптивный по сессии, ограничен сверху):

- **merge_tol** считается **внутри каждой сессии** по zone_widths этой сессии; ограничение сверху: **merge_tol = min(..., 1.0)** (**LEVEL_MERGE_TOL_SESSION_CAP = 1.0**).

**Условие добавления зоны в группу (anchor-based, без chain-effect):**

1. |zone_mean − **anchor_mean** группы| ≤ merge_tol (сравнение с фиксированным центром группы, не с расширенным union).  
2. Сильное пересечение с **seed-зоной** группы (первая зона в группе): пересечение покрывает не менее **80%** ширины меньшей зоны (min_overlap_ratio=0.8).

Границы группы в процессе не раздуваются; решение «добавить в группу» принимается только по anchor и seed. В конце для каждой группы выбирается **representative** — исходная зона, ближайшая по центру к медиане центров группы.

**Порог пересечения (session-dedupe):** min_overlap_ratio = **0.8**. Cap merge_tol: **LEVEL_MERGE_TOL_SESSION_CAP = 1.0**.

После дедупликации возвращается список **(zlo, zhi, weight)** — по одному представителю на группу, **weight** = число зон в группе (len(items)).

**Опционально: зоны ES overnight** (источник `ES_1min_90d.csv`): это отдельный набор зон для DBSCAN, рассчитанный **по последним 10 SPX датам** (те же даты, что и последние 10 SPX RTH сессий в `dates`). Для каждой SPX‑сессии \(d\) берём окно:

- **00:00 ≤ t < 09:30 NY** на дату \(d\) (09:30 не включаем).
- Берём **последние N_PREMARKET_SESSIONS = 10** дат из SPX history window (**последние 10 дат из `dates`**).

Далее те же L1/L2 и тот же **квантильный trim q10/q90** границ зоны (см. выше); диапазон фильтра: **pm_lo/pm_hi = open_pred ± mean_range_spx × PREMARKET_RANGE_MULT** (**PREMARKET_RANGE_MULT = 1.1**); в ES: + basis_d.

**Basis для каждой overnight‑сессии ES (на дату \(d\)) считается отдельно и применяется только к зонам этой даты. Ключевой момент: `prev_session` определяется по SPX календарю** (по списку SPX торговых дат), чтобы корректно перескакивать выходные/праздники. Базис берётся по предыдущей RTH‑сессии (prev_session):

- `basis_d = Close_last_ES(prev_session) − Close_last_SPX(prev_session)`
- где `prev_session` — предыдущая SPX торговая дата относительно \(d\) (по списку SPX дат, то есть для понедельника prev_session обычно пятница).

Конвертация **spx_zone = es_zone − basis_d**.

Чтобы зоны одного overnight ES не дублировались, выполняется **дедупликация зон внутри каждой overnight‑сессии** (аналогично SPX session-dedupe, но отдельно по каждой дате \(d\)) **до DBSCAN**. Веса для DBSCAN: **weight = group_size × PREMARKET_ZONE_WEIGHT**.

**Отдельно (не premarket-zones): два ES overnight уровня High/Low** добавляются **после этапа E** как дополнительные финальные уровни:

- **Окно**: берём бары ES за дату **plan_for** в интервале **00:00 ≤ t < 09:30 NY** (09:30 не включаем).
- **Базис (basic)**: `basic = Close_last_ES(prev_session) − Close_last_SPX(prev_session)` (предыдущая RTH‑сессия относительно plan_for). Знак сохраняется (это не модуль).
- **Перевод в SPX-эквивалент**:
  - `High_level_SPX = High_ES − basic`
  - `Low_level_SPX  = Low_ES  − basic`
Примечание: в текущей логике базис для High/Low ES берётся по **RTH close предыдущей сессии** (см. формулу выше) и **не масштабируется по VIX**.

---

## 3. Кластеризация DBSCAN

Цель: объединить близкие по цене zone_means в кластеры и отобрать самые «насыщенные» кластеры как кандидаты в уровни.

### 3.1 Параметры (адаптивные)

- **Метрика**: евклидова (по одному признаку — цена zone_mean).
- **eps** — максимальное расстояние между двумя точками в одном кластере (п.п.); вычисляется **по данным**, а не задаётся фиксированно.
- **min_samples** — минимальное число точек в кластере; задаётся **долей от числа зон**, а не абсолютной константой.

**Вычисление eps:**

1. Отсортировать **zone_means**, вычислить **diffs** = разности соседних значений (np.diff).
2. Среди строго положительных разностей взять квантиль 0.25: **eps_raw** = quantile(diffs[diffs > 0], 0.25).
3. **eps** = clip(eps_raw × 1.5, **LEVEL_DBSCAN_EPS_CLIP_LO**, **LEVEL_DBSCAN_EPS_CLIP_HI**), затем **eps = max(eps, 0.9)** (нижняя граница 0.9 п.п.).
4. Если положительных разностей нет (или данных мало), запасной вариант: **eps** = clip(mean_range_spx × 0.01, 0.9, 3.0).

**Вычисление min_samples:**

- **n** = число зон после дедупликации.
- **min_samples** = clip(round(n × 0.03), **4**, **16**).

Таким образом, при 120 зонах min_samples ≈ 4 (нижняя граница), при 500+ зонах — 16 (верхняя граница).

### 3.2 Выполнение

- Вход: **zone_means** (N,) и **zone_weights** (N,) — веса = размер группы при session-merge; для DBSCAN: **X** формы (N, 1), `fit(X, sample_weight=zone_weights)`.
- Каждой точке присваивается **label**: ≥ 0 — номер кластера, −1 — шум.

### 3.3 Отбор топ-кластеров

- Кластеры (с номерами ≥ 0) группируются по **label**.
- Сортировка: по **убыванию размера** кластера (число точек).
- Берутся первые **LEVEL_TOP_N_CLUSTERS = 10** кластеров (может быть меньше, если кластеров меньше 10).

---

## 4. Формирование уровня по кластеру (перцентили)

Для каждого из отобранных топ-кластеров уровень считается не как простое среднее по кластеру, а по **среднему значений в перцентильном диапазоне**, чтобы уменьшить влияние выбросов.

### 4.1 Перцентильный диапазон

- **LEVEL_CLUSTER_PCT_LO = 40.0** — нижняя граница (40-й перцентиль).
- **LEVEL_CLUSTER_PCT_HI = 60.0** — верхняя граница (60-й перцентиль).

Для кластера с массивом **vals** (zone_means этого кластера):

- `p40 = percentile(vals, 40)`
- `p60 = percentile(vals, 60)`
- В полосу **[p40, p60]** попадают значения «без крайних» (обрезаем крайние 40% с каждой стороны).

### 4.2 cluster_mean

- Берутся все **vals** в диапазоне `[p40, p60]`.
- **cluster_mean** = среднее по этим значениям.
- Если в полосе ни одной точки нет, **cluster_mean** = среднее по всем **vals** кластера.

Один кластер → один уровень-кандидат **cluster_mean**.

---

## 5. Фильтрация по полосе вокруг open_pred

Оставляются только те **cluster_mean**, которые попадают в полосу:

- **margin_lo** = open_pred − mean_range_spx − **LEVEL_FILTER_MARGIN**
- **margin_hi** = open_pred + mean_range_spx + **LEVEL_FILTER_MARGIN**

Константа: **LEVEL_FILTER_MARGIN = 5.0** (п.п.).

Условие: **margin_lo ≤ cluster_mean ≤ margin_hi**.

Пример: open_pred = 6951, mean_range_spx = 79.6 → полоса **6866.4 – 7035.6**. Уровни вне неё отбрасываются.

---

## 6. Разделение на Support и Resistance

- **Support** — все отфильтрованные уровни **строго ниже** open_pred, упорядочены по убыванию (ближайший к open_pred первый).
- **Resistance** — все отфильтрованные уровни **строго выше** open_pred, упорядочены по возрастанию.

Уровень, равный open_pred, в текущей логике не попадает ни в support, ни в resistance.

Для этапа E каждый кандидат хранится как пара **(cluster_mean, cluster_weight)** (суммарный вес зон в кластере).

---

## 7. Этап E: объединение близких итоговых уровней (level_dedup_tolerance)

После DBSCAN и фильтрации получаем список **кандидатов в уровни** (cluster_mean + суммарный вес кластера). Топ кластеров отбирается по **убыванию суммарного веса** (не по числу точек). Это не объединение сырых зон, а объединение уже **финальных уровней-кандидатов**, чтобы не отдавать пользователю два почти совпадающих уровня (например 6948.4 и 6949.1).

**Условия объединения (одновременно):**

- **(а)** Уровни на **одной стороне** от open_pred: support объединяются только с support, resistance только с resistance (никогда не смешивать support и resistance).
- **(б)** Уровни достаточно близко: **|L₂ − L₁| ≤ tol**, где **tol** — эффективный допуск (см. ниже).

Сравниваются **соседние** элементы в уже отсортированном списке: support — по убыванию, resistance — по возрастанию.

**Итоговый merged level (взвешенное среднее по размеру кластера):**

Если уровень 6948.4 получен из кластера размера 40, а 6949.1 — из кластера размера 12, то объединённый уровень:

**merged = (6948.4×w1 + 6949.1×w2) / (w1 + w2)** (по весам кластеров).

То есть кластер с большим суммарным весом сильнее влияет на итоговую цену.

**Эффективный допуск (адаптивный):** **tol = max(2.0, mean_range_spx × 0.06)**. При большом среднем диапазоне дня merge усиливается.

---

## 8. Итоговая схема по этапам A–F

Чтобы не смешивались разные типы merge, используется следующая последовательность.

| Этап | Описание |
|------|----------|
| **A** | **Поиск raw zones** — из каждой сессии ищутся L1/L2-зоны. Результат: список пар (zlo, zhi). |
| **B** | **Merge зон внутри одной сессии** — anchor-based: \|zm − anchor_mean\| ≤ merge_tol, overlap с seed ≥ 80%; merge_tol cap 1.0. Возврат (zlo, zhi, weight), weight = len(items); DBSCAN с sample_weight=zone_weights. |
| **C** | **DBSCAN** по всем zone_means после этапа B — собрать похожие зоны в кластеры по всей истории. |
| **D** | **Один representative level на кластер** — кластеры сортируются по **суммарному весу**; для каждого считается cluster_mean (перцентили 40–60), фильтр по полосе. Сохраняем (cluster_mean, cluster_weight). В диагностику по каждому кластеру выводятся **cluster_weight_total**, **cluster_weight_rth**, **cluster_weight_pm**. |
| **E** | **Merge близких итоговых уровней** — внутри support и внутри resistance: если два соседних уровня ближе чем **tol** (max(2.0, mean_range_spx×0.06)), объединяем в один по взвешенному среднему по cluster_weight. |
| **F** | **Добавление ES overnight High/Low** — два уровня (High/Low) из ES за **00:00–<09:30 NY** на plan_for, переведённые в SPX-эквивалент через **basic**; затем дедупликация и сортировка финальных списков support/resistance. |

Пошагово:

1. Загрузка **open_pred** и **mean_range_spx**; задание **level_range_spx** = open_pred ± mean_range_spx × 1.25.
2. **A:** По каждой сессии — сбор зон L1 и L2, только зоны, пересекающиеся с level_range_spx.
3. **B:** Anchor-based merge по SPX RTH; выход (zlo, zhi, weight). Без раннего выхода при пустом merged_zones. Опционально: ES premarket — даты из **dates**, пересечение с ES premarket, последние **N_PREMARKET_SESSIONS**; L1/L2, полоса **PREMARKET_RANGE_MULT**; ES→SPX по basis_d; вес **PREMARKET_ZONE_WEIGHT**; объединение **zone_means** = [zone_means_rth, pm_means], **zone_weights** = [zone_weights_rth, pm_weights].
4. **C:** Адаптивные eps и min_samples; **DBSCAN** по zone_means с sample_weight=zone_weights; топ-10 кластеров.
5. **D:** Кластеры по убыванию суммарного веса; для каждого cluster_mean (перцентили 40–60), фильтр по полосе; список **(cluster_mean, cluster_weight)**; разделение на support и resistance.
6. **E:** Merge соседних уровней при \|L₂−L₁\| ≤ tol, tol = max(2.0, mean_range_spx×0.06) (weighted mean по cluster_weight).
7. **F:** Добавляем два ES overnight уровня (High/Low) за **00:00–<09:30 NY** на plan_for (см. раздел 2.4), распределяем по сторонам относительно open_pred, затем **dedupe+sort**.
8. Итоговые списки **Support** и **Resistance** передаются в **build_TradePlan.py**: Summary (Support Levels / Resistance Levels) и лист Trade Plan (Key levels по сценариям).

---

## 9. Основные константы (сводка)

| Константа | Значение | Назначение |
|-----------|----------|------------|
| LEVEL_RANGE_MULT | 1.25 | Множитель к mean_range_spx для диапазона поиска зон |
| LEVEL_MERGE_TOL_SESSION_CAP | 1.0 | Верхняя граница merge_tol внутри сессии (п.п.) |
| LEVEL_DBSCAN_EPS_CLIP_LO | 0.9 | Нижняя граница для адаптивного eps (п.п.); дополнительно eps = max(eps, 0.9) |
| LEVEL_DBSCAN_EPS_CLIP_HI | 3.0 | Верхняя граница для адаптивного eps (п.п.) |
| LEVEL_DBSCAN_EPS_DEFAULT | 1.0 | Запасное значение eps (при отсутствии данных) |
| LEVEL_DBSCAN_MIN_SAMPLES_DEFAULT | 25 | Запасное значение min_samples |
| LEVEL_TOP_N_CLUSTERS | 10 | Максимум кластеров для уровней |
| LEVEL_FILTER_MARGIN | 5.0 | Допуск при фильтрации уровней (open_pred ± mean_range_spx ± 5) |
| LEVEL_CLUSTER_PCT_LO | 40.0 | Нижний перцентиль для расчёта уровня по кластеру |
| LEVEL_CLUSTER_PCT_HI | 60.0 | Верхний перцентиль для расчёта уровня по кластеру |
| N_PREMARKET_SESSIONS | 10 | Число ES overnight-сессий для уровней (даты = последние N из `dates`) |
| PREMARKET_ZONE_WEIGHT | 2.0 | Вес премаркет-зон в DBSCAN (чтобы не теряться на фоне RTH) |
| PREMARKET_RANGE_MULT | 1.1 | Множитель к mean_range_spx для полосы pm_lo/pm_hi (ES overnight 00:00–<09:30 NY) |

**Параметры:** L1–L2 см. в разделе 2.1. **Этап B:** anchor-based, merge_tol cap 1.0, overlap ≥ 0.8, возврат (zlo, zhi, weight); DBSCAN с sample_weight. min_samples = clip(round(n×0.03), 4, 16). **Этап D (диагностика):** по каждому кластеру — cluster_weight_total, cluster_weight_rth, cluster_weight_pm. **Этап E:** tol = max(2.0, mean_range_spx×0.06); объединение соседних итоговых уровней на одной стороне от open_pred, weighted mean по cluster_weight.

# `build_TradePlan.py` — актуальное описание модуля

Документ описывает **текущую** реализацию [`build_TradePlan.py`](../build_TradePlan.py): сборка Excel-книги (**Summary** + **Trade Plan**) для однодневного сценария SPX. При расхождениях источник истины — код.

---

## 1. Зависимости

### Прямые импорты

- **`open_price`**: `predict_spx_open`, пороги `VIX_REGIME_LOW`, `VIX_REGIME_NORMAL_HI`, `VIX_REGIME_HIGH_HI`.
- **`levels`**: `predict_key_levels`, `KeyLevels`, `level_search_band_from_scenario_targets`.
- **`bias`**: `predict_bias_and_probability`, `format_bias_console`.
- **`chop`** (опционально, `try` / `ImportError`): `predict_session_chop` как `chop_predict_session`, `load_chop_result` как `chop_load_result`.
- **`trigger`**: `trigger.compute_opening_trigger(...)`.

### Чего нет в модуле

- Прямого импорта **`entry_target`**.
- Прямого импорта **`trade_plan_calibration`** (режим VIX для Summary задаётся через `open_price`).

---

## 2. Жёсткий порядок колонок Trade Plan

Константа **`TRADE_PLAN_HEADERS`** задаёт столбцы листа **Trade Plan** (строго в этом порядке):

`Scenario`, **Chop**, `Bias Trend`, `Opening trigger`, `Expected Range`, `Key levels`,  
`Entry point_1` … `Target3`, **`Entry point_4`**, **`Target4`**, **`Max Target (data history)`**, **`Bar sequence`**, **`Bar middle range`**, **`Risky zone (NO TRADE)`**.

Колонка **Chop** заполняется многострочным текстом по сегментным прогнозам chop (`predicted_chop_pct_1`…`_4`), если они есть; иначе — суммарным `predicted_chop_pct`.  
**Bar sequence** и **Bar middle range** формируются из **`ChopSessionResult.similarity_sequence_range`** через **`format_similarity_sequence_range_trade_cells`** (пулы bull/bear из chop).

---

## 3. Пайплайн `main()`

1. Разрешение каталога данных (`_choose_data_dir`), загрузка **ровно заявленных в шапке файла** CSV (дневные и минутные ряды SPX/ES/VIX/…).
2. **`plan_for`** = календарная дата из **`--expiry`** (аргумент назван legacy «expiry», фактически это **дата сессии плана**).
3. **`ref_date` / ref_close** по последнему дню SPX daily до `plan_for`; минутный SPX нарезается в RTH-слайс.
4. **`predict_spx_open(...)`** — единственный источник **`open_pred`** и **`ref_close`** для якоря плана (пути ES 5m/15m/1m/RTH, VIX 5m/15m/1m, VIX9D/VIX3M 5m, дневной VIX — как в вызове в `main`).
5. **`predict_bias_and_probability`** — текстовые зоны bias и числовые скоры для сценариев / Summary.
6. **Chop**: сначала попытка **`chop_load_result(plan_for, …)`**; при отсутствии файла — **`chop_predict_session(...)`** с переданным `open_pred` и `spx_1m`.
7. При наличии сохранённого **`levels_log_<plan_for>.json`** (корень проекта или data dir) уровни могут подставляться из EM-пайплайна без повторного **`predict_key_levels`** — см. вызов `assemble_trade_plan` в `main`.
8. **`assemble_trade_plan(...)`** → строки Trade Plan + **`plan_meta`**.
9. Запись Excel (Summary синхронизируется с **`plan_meta`**).

---

## 4. `assemble_trade_plan(...)` — логика

### 4.1 Якорь открытия и EM

- **`open_pred`** = **`open_pred_override`**, если число конечно; иначе **`ref_close`** (вызов из `main` всегда подставляет рассчитанный open из `open_price`).
- Базовый **`em_pts`**: из аргумента; затем при наличии chop перезаписывается по приоритету:
  1. если заданы и **`mean_range_spx_bull`**, и **`mean_range_spx_bear`** оба > 0 → **`em_pts = max(bull, bear)`**;
  2. иначе любой один положительный из них;
  3. иначе **`mean_range_spx`** аргумента;
  4. если при этом раздельные bull/bear отсутствуют и в chop есть **`em_adjusted`** — подстановка единого EM и пересчёт масштабов сценариев.

### 4.2 Opening trigger

- **`trigger.compute_opening_trigger(open_pred=open_pred)`** → текст для колонки **Opening trigger**.

### 4.3 Масштаб ожидаемого хода по сценарию

- **`em_bull_base` / `em_bear_base`** берутся из раздельных chop EM или из общего **`em_pts`**.
- **`em_bull = scale_expected_move(..., "Bullish")`**, **`em_bear = scale_expected_move(..., "Bearish")`** — для Summary и связанных подписей.

### 4.4 Chop в таблице и meta

- Из **`chop_result`** читаются **`predicted_chop_pct`**, **`predicted_chop_pct_1`…`_4`**, диагностика.
- В **`plan_meta["chop_duration_predictions"]`** записываются ключи **`predicted_chop_pct_k`** для k=1…4 (числа или NaN).
- **`chop_duration_summary_text`** / ячейка **Chop** — человекочитаемые строки по сегментам (подписи временных окон совпадают с bias-сегментами в других местах модуля).

### 4.5 Ключевые уровни

- Если переданы **`key_levels_from_em`** (и опционально **`levels_diag_from_em`**) — создаётся **`KeyLevels`** без нового вызова **`predict_key_levels`**.
- Иначе формируется опциональная полоса поиска **`level_search_band_from_scenario_targets(...)`** из **`target_bullish_*` / `target_bearish_*`** chop, затем **`predict_key_levels`** с **`level_search_lo/hi`**, **`mean_range_spx`**, **`open_pred`**, **`n_sessions=15`**, **`n_each=5`**.

### 4.6 Expected Range (две строки сценариев)

Приоритет расчёта границ **`bull_lo/hi`**, **`bear_lo/hi`**:

1. **Если в chop есть раздельные положительные `mean_range_spx_bull` и `mean_range_spx_bear` и все четыре `mean_ratio_*` конечны:**  
   - \(O =\) `open_pred`  
   - **Bullish:** `bull_hi = O + mr_bull * mean_ratio_bull_high`, `bull_lo = O − mr_bull * mean_ratio_bull_low`  
   - **Bearish:** `bear_hi = O + mr_bear * mean_ratio_bear_high`, `bear_lo = O − mr_bear * mean_ratio_bear_low`  
   Множители **1.1** из старых описаний **здесь не используются** — только произведение EM стороны на коэффициент из chop.

2. **Иначе**, если доступны единый **`mr_for_er`** (из `mean_range_spx` результата chop, аргумента или **`em_adjusted`**) и те же **`mean_ratio_*`**:  
   - **Bullish:** `bull_hi = O + mr * r_bull_hi`, `bull_lo = O − mr * r_bull_lo`  
   - **Bearish:** `bear_hi = O + mr * r_bear_hi`, `bear_lo = O − mr * r_bear_lo`

3. **Фолбэк на долю EM:** при конечном **`em_for_er`** > 0:  
   - Bullish: от **`O − 0.10·em_for_er`** до **`O + 1.00·em_for_er`**  
   - Bearish: от **`O − 1.00·em_for_er`** до **`O + 0.10·em_for_er`**

4. **Жёсткий фолбэк:** **`em_fallback = 50`** пунктов с теми же пропорциями 10%/100%.

Дополнительно для Summary может вычисляться **`em_quantile_0_9`** по истории дневных диапазонов SPX (≥60 точек) — используется в других подписях, но не заменяет цепочку выше для самих границ Expected Range в **`assemble_trade_plan`**.

### 4.7 Chop zones и Risky zone

- Зоны приводятся к парам **`(zlo, zhi)`** и фильтруются по пересечению с интервалом Expected Range каждого сценария (**`_chop_zones_by_scenario_range`**).
- Колонка **Risky zone** заполняется из этих списков (не из entry/target логики).

### 4.8 Entry / Target: `_build_entries_targets_levels`

Внутренняя машина состояния строит **четыре пары** **`entry1/target1` … `entry4/target4`** для каждого сценария:

- Старт: bull — якорь **`open_pred + 3`**, направление «вверх»; bear — **`open_pred − 3`**, «вниз».
- Уровни берутся только из списков поддержек/сопротивлений **внутри** Expected Range сценария (отбор **`bull_above/below`**, **`bear_above/below`** в `assemble_trade_plan`).
- Минимальное разделение цели от точки входа по уровню: **`ENTRY_TARGET_LEVEL_MIN_SEP_PTS = 10`**.
- Если следующий уровень в направлении движения ближе **10** пунктов к уже выбранному target — сдвиг следующего entry на **±1.5** пункта относительно модели «на уровень выше/ниже»; иначе шаг **±2.0** от текущего target.
- **Разворот у границы сценария:** если target оказывается в **10** пунктах от **`hi`/`lo`**, направление меняется на противоположное; опционально используется шаг к **«Max Target (data history)»** из chop (**`target_bullish_high`** для bull-сценария, **`target_bearish_low`** для bear) с отложенным разворотом (**`defer_flip_after_max`**), чтобы не «ломать» последовательность при наличии более дальнего исторического таргета.
- После разворота допускается проход по «своим» уровням в обратном порядке (**`_next_*_on_path`**) и использование уровней противоположной стороны при необходимости.
- Уровни, уже бывшие target, могут исключаться через **`used_targets`** (ключ — округление до 2 знаков), кроме режима retracement по пути разворота.

### 4.9 Форматирование Key levels в строке

- **`_levels_txt_for_scenario`**: список Key levels формируется **внутри диапазона Expected Range** и отображается с подписью ранга из **`levels_log`** или меток **High ES / Low ES** при совпадении с overnight ES из диагностики.
  - **Bullish**: уровни сортируются **по возрастанию** (низ → верх).
  - **Bearish**: уровни сортируются **по убыванию** (верх → низ).

---

## 5. `plan_meta` (ключевые поля)

Помимо **`open_pred`**, **`em_pts`**, масштабированных EM по сторонам, границ Expected Range:

- **`chop_zones`**, **`chop_zones_count`**, **`is_chop_day`** (итог: **`≥3` зон или эвристика `predict_chop_day`**),
- **`chop_duration_predictions`** — словарь с **`predicted_chop_pct_1`…4**,
- **`chop_duration_summary_text`**,
- **`total_chop_duration_pct`**,
- **`target_bullish_*`**, **`target_bearish_*`**,
- **`chop_diagnostics`** (копия словаря из chop).

Поле **`chop_duration_predictions` в текущей версии заполняется** (в отличие от старых заготовок «пустой dict»).

---

## 6. Пограничные случаи

- Модуль **chop** может отсутствовать: план строится на фолбэках EM / диапазонов и без chop-колонок по данным chop.
- **`mean_range_spx_bull` / `_bear`** из chop позволяют задать **асимметричный** Expected Range без ручных множителей 1.1 в коде.

---

## 7. Согласованность с кодом

- Описание отражает **динамический Trade Plan** с **4-мя** шагами Entry/Target, колонками **Chop**, **Max Target (data history)** и **sequence+range** из chop.
- Формулы Expected Range приведены в соответствие с ветвлением **`assemble_trade_plan`** (раздельные MR bull/bear против единого `mr_for_er` против EM-фолбэка).

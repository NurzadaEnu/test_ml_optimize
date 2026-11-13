# Как запустить пайплайн — все этапы по порядку

## Требования

```bash
pip install -r requirements.txt
```

## Этап 1 — Генерация данных (`generate_data.py`)

Синтетический датасет (3000 строк):
- **Объём:** `y1 = x1·η(x2,x3)`, η откалибрована МНК по трём режимам таблицы 1; η ≤ 0.995, поэтому `y1 < x1`.
- **Качество:** μ1–μ3 — степени превращения (реактор вытеснения 1-го порядка, кинетика Аррениуса), откалиброванные в центре рабочих диапазонов (x2 = 27.5, x3 = 321.5, x4 = 2.75, x5 = 350). Решение AorRA+PO в генераторе не используется.

```bash
python3 generate_data.py
```

Ожидаемый вывод:
```
deterministic       : y1_table=76.0  y1_model=76.00
AorRA+PO (fuzzy)    : y1_table=77.5  y1_model=77.50
real_production     : y1_table=77.0  y1_model=77.00
```

## Этап 2 — Обучение (`train_model.py`)

4 модели XGBoost (n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9), разбиение 80:20. Печатает RMSE, R², MAE на тесте → `data/test_metrics.csv`, графики `plots/parity_plots.png`, `plots/feature_importance.png`.

## Этап 3 — Валидация (`validate_model.py`)

- 5-fold CV: R² по фолдам, mean ± std → `data/cv_metrics.csv`. Синтетические выборки независимы (нет автокорреляции), поэтому случайный K-fold корректен.
- Для реальных данных — `cross_validation(df, time_ordered=True)` (TimeSeriesSplit).
- Сверка суррогата с генератором на режимах таблицы 1.

## Этап 4 — Оптимизация (`optimize.py`)

Единая задача (Eq. 16): `max y1` при `Score ≥ 0.90` и `y1 ≤ x1`, случайный поиск по 60 000 точкам на суррогате.
- проверка рекомендованного режима на генераторе;
- `data/table1_common_model.csv` — все режимы на одной модели качества;
- `plots/fig5_sensitivity_front.png` — (a) y1 и Score vs температура, (b) фронт ε-constraint (`data/eps_front.csv`);
- проверка утечки: оптимизация при x2 = 28 (как у AorRA+PO).

## Сравнение алгоритмов (`compare_models.py`)

8 алгоритмов, 5-fold CV → `data/model_comparison.csv`, `plots/model_comparison.png`.

## Полный запуск

```bash
python3 generate_data.py && python3 train_model.py && python3 validate_model.py && python3 optimize.py
```

## Реальные данные завода

Замените `data/synthetic_dataset.csv` (те же колонки: `x1_syre, x2_davlenie, x3_temperatura, x4_skorost, x5_vsg, y1_obyem, mu1_nepredelnye, mu2_sera, mu3_kisloty`), запустите этапы 2–4, используйте временной сплит и сузьте `BOUNDS` до исторических диапазонов.

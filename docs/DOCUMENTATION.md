# Документация PLER-2.0 Experiment

Полное описание решения для сравнения качества 3D-моделей: назначение, архитектура, использование и разработка.

---

## 1. Назначение

**PLER-2.0 Experiment** — исследовательский инструмент для:

1. **Объективной оценки** качества 3D-мешей (формат OBJ) относительно эталона.
2. **Сравнения метрик** между собой и с субъективными оценками **MOS** (Mean Opinion Score).
3. **Воспроизведения экспериментов** на эталонном датасете QV-сфера или на пользовательских данных.
4. **Генерации контролируемых деградаций** для проверки чувствительности метрик.

### Зачем это нужно

При разработке и валидации метрик качества 3D-контента (стриминг, сжатие, LOD) необходимо:

- понимать, насколько метрика коррелирует с человеческим восприятием (MOS);
- сравнивать метрики с разными шкалами и направлениями («меньше = лучше» vs «больше = лучше»);
- иметь воспроизводимый пайплайн от данных до отчёта.

Данное решение объединяет расчёт, нормировку, визуализацию и экспорт в одном Streamlit-приложении.

---

## 2. Как работает система

### 2.1. Общий пайплайн

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Датасет    │────▶│  Пары (ref,dist) │────▶│  Расчёт метрик  │
│  QV / свои  │     │  + ref vs self   │     │  metrics_core + │
└─────────────┘     └──────────────────┘     │  external_metrics│
                                              └────────┬────────┘
                                                       │
                       ┌───────────────────────────────┼───────────────────────────────┐
                       ▼                               ▼                               ▼
              ┌────────────────┐              ┌─────────────────┐              ┌──────────────┐
              │ Сырые таблицы  │              │ Нормировка 0-100│              │ Корреляции   │
              │ CSV / XLSX     │              │ якорь ref=self  │              │ с MOS        │
              └────────────────┘              └─────────────────┘              └──────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Графики Plotly  │
                                              │ + MOS           │
                                              └─────────────────┘
```

### 2.2. Пары сравнения

Для каждой искажённой модели создаётся пара:

- **reference** — путь к эталонному OBJ
- **distorted** — путь к искажённому OBJ

После обработки всех пар функция `append_reference_self_pairs()` добавляет для каждого уникального эталона пару **(ref, ref)** с меткой `distorted_model = __REFERENCE_SELF__`.

### 2.3. Якорная нормировка

Функция `normalize_with_ref_anchor()` в `results_analysis.py`:

| Тип метрики | 100 (лучшее) | 0 (худшее) |
|-------------|--------------|------------|
| higher is better | значение ref vs self | min среди искажённых |
| lower is better | значение ref vs self | max среди искажённых |

Формула (higher is better):

```
norm = 100 × (v − v_worst) / (v_ref_self − v_worst)
```

Результат обрезается в [0, 100]. Это даёт **общую шкалу** для Chamfer, PLER, MSDM2 и др. на одном графике.

**MOS_norm**: `100 × MOS / MOS_ref`, где MOS_ref — MOS эталона (или max MOS в группе).

### 2.4. Checkpoint и возобновление

При расчёте после каждой пары результаты пишутся в `results/checkpoint_<run_id>.csv`. При повторном запуске с тем же `run_id` уже обработанные пары пропускаются.

---

## 3. Режимы для пользователя

### 3.1. QV-сфера (встроенная)

**Данные:** `data/qv_sphere/Sphere_*.obj`, эталон — `Sphere_14.obj`.

**MOS:** `data/mos.csv` — transposed CSV:

```csv
model_name;Sphere_3;Sphere_4;...
mos;1.25;1.5;...
```

**Шаги:**
1. Тип датасета → QV-сфера
2. Источник → Встроенный Sphere
3. Выберите метрики
4. Запустить расчёт

### 3.2. QV-сфера (свои модели)

Для пользователей с готовыми искажёнными OBJ.

**Шаги:**
1. QV-сфера → Загрузить свои OBJ
2. Загрузите эталон и несколько искажённых файлов
3. Укажите порядок (1 = худшее качество, n = лучшее):

```
worst.obj, medium.obj, good.obj
```

или

```
1:Sphere_1.obj, 2:Sphere_5.obj, 3:Sphere_12.obj
```

4. Опционально — CSV с MOS
5. Запустить расчёт

**Порядок важен** для оси X «Номер модели» и интерпретации `degradation_param`.

### 3.3. Свои модели (генерация деградаций)

**Данные:** `data/high_quality/*.obj` (bunny, teapot, suzanne, …)

**Методы:**
| Метод | Параметр | Описание |
|-------|----------|----------|
| Decimation | ratio 0..1 | Доля сохраняемых граней |
| Noise | sigma | Амплитуда шума вершин |
| Smoothing | iterations | Сглаживание Laplacian |
| Combined | dec + noise | Комбинация |

Результаты: `generated/<run_id>/<model>/`.

---

## 4. Метрики

### 4.1. Встроенные (`metrics_core.py`)

| Метрика | Направление | Описание |
|---------|-------------|----------|
| PLER-2.0 (complex) | ↑ | Комплексная метрика PLER |
| PLER geometry | ↑ | Геометрическая компонента (dB) |
| TSI | ↑ | Topology Stability Index |
| AAD | ↓ | Average Angular Deviation |
| Chamfer (mm) | ↓ | Симметричный Chamfer |
| Hausdorff (mm) | ↓ | Hausdorff distance |
| F-score @ τ | ↑ | F-score при порогах 0.01–0.1 |
| PSNR (dB) | ↑ | PSNR точечного облака |

### 4.2. Внешние (`external_metrics.py`)

| Метрика | Требования |
|---------|------------|
| MSDM2, PCQM, 3D-PSSIM | Упрощённые реализации |
| PointPCA+, PQI, GeodesicPSIM | scikit-learn / trimesh |
| CMDM | Vertex colors (иначе NaN) |
| **HybridMQA** | Checkpoint + PyTorch |
| **FMQM** | OBJ + PNG текстуры |

### 4.3. HybridMQA / FMQM

**Установка:**

```bash
python scripts/setup_ml_metrics.py
```

Структура после установки:

```
third_party/hybridmqa/     — репозиторий
third_party/fmqm/          — репозиторий
models/checkpoints/ckpt_TMQA_adapted.pth
```

В UI укажите путь к checkpoint. Переменная окружения `HYBRIDMQA_CKPT` также поддерживается.

**FMQM:** положите `model.png` рядом с `model.obj` или в `data/textures/<stem>/`.

---

## 5. Интерфейс результатов

### Вкладка «Сырые метрики»

Исходные значения, метаданные: `model_number`, `n_points`, `model_weight_mb`, `MOS`, `is_reference_self`.

Строка ref vs self отображается как **«эталон (ref=self)»**.

### Вкладка «Нормированные»

Шкала 0–100 по якорю. Колонка `MOS_norm` при наличии MOS.

### Вкладка «Корреляции»

PLCC, SROCC, Kendall τ с MOS. Строка ref vs self **исключается** из расчёта.

### Вкладка «Графики»

- Ось X: номер модели, n_points, вес, параметр деградации
- Масштаб linear / log
- **MOS** доступен в multiselect (отображается как MOS_norm 0–100)

### Вкладка «Корр. матрица»

Spearman между выбранными метриками (heatmap).

---

## 6. Руководство для разработчиков

### 6.1. Структура модулей

| Модуль | Ответственность |
|--------|-----------------|
| `app.py` | UI Streamlit, оркестрация |
| `pair_utils.py` | Построение пар, ref vs self, upload QV |
| `metrics_core.py` | PLER, классические метрики |
| `external_metrics.py` | Внешние метрики, EXTERNAL_COMPUTE_MAP |
| `ml_metrics.py` | Subprocess HybridMQA/FMQM |
| `results_analysis.py` | Нормировка, корреляции, enrich |
| `degradation.py` | DegradationGenerator |
| `obj_helpers.py` | Утилиты OBJ |

### 6.2. Добавление новой метрики

1. Реализуйте функцию `compute_<name>(ref_path, dist_path) -> float` в `external_metrics.py` или `metrics_core.py`.
2. Добавьте имя в `METRIC_DIRECTION` и `METRIC_THEORETICAL_BOUNDS` (`metrics_core.py`).
3. Зарегистрируйте в `EXTERNAL_COMPUTE_MAP` или внутреннем dispatch `compute_selected_metrics`.

Якорная нормировка подхватит направление автоматически из `METRIC_DIRECTION`.

### 6.3. API нормировки

```python
import results_analysis as ra

# Якорная (по умолчанию в UI)
norm = ra.normalize_with_ref_anchor(df, ["Chamfer (mm)", "MSDM2"])

# Теоретическая
norm = ra.normalize_theoretical(df, metric_cols)

# Таблица для UI
table = ra.build_display_table(df, metric_cols, normalized=True, normalization="anchor")
```

### 6.4. API пар

```python
from pair_utils import append_reference_self_pairs, build_custom_qv_pairs_from_uploads

pairs = [...]  # ref/dist dicts
pairs = append_reference_self_pairs(pairs)
```

### 6.5. Важные замечания по коду

- **Не называйте модули `analysis.py` или `mesh_utils.py`** — конфликты с jedi/кэшем.
- **`clear_pycache()`** вызывается при старте `app.py`.
- **pymeshlab decimation:** используйте `targetfacenum`, не `targetperc` (2025.x).
- **Большие меши:** KDTree субсэмплинг в `metrics_core.MAX_KDTREE_POINTS`.

### 6.6. Тестирование

```bash
pytest tests/test_integration.py -v
```

Покрытие:
- ref vs self pairs
- parse order
- anchor normalization (higher/lower)
- MOS normalization
- correlation filter

### 6.7. Запуск в dev

```bash
streamlit run app.py --server.runOnSave true
```

Логи метрик — в stdout (logging + tqdm).

---

## 7. Форматы данных

### OBJ

Wavefront OBJ с вершинами `v x y z` и гранями `f ...`. Текстуры и материалы нужны только для FMQM/CMDM.

### MOS CSV

**Вариант A** (transposed, как `data/mos.csv`):

```csv
model_name;ModelA;ModelB
mos;3.5;4.2
```

**Вариант B** (long):

```csv
model_name;mos
ModelA;3.5
ModelB;4.2
```

### Результаты

`results/results_<run_id>.csv` — все пары включая ref vs self.

Колонки: meta + выбранные метрики + `compute_time_s`, `error`.

---

## 8. Troubleshooting

| Проблема | Решение |
|----------|---------|
| ImportError analysis | Перезапустите Streamlit; проверьте отсутствие `analysis.py` |
| Decimation не меняет mesh | Обновите `degradation.py`; ratio через targetfacenum |
| HybridMQA NaN | `python scripts/setup_ml_metrics.py`, проверьте checkpoint |
| FMQM NaN | Добавьте PNG текстуры рядом с OBJ |
| PLER очень долго | Уменьшите число пар или отключите PLER |
| График пустой | Проверьте NaN в метриках; выберите другую ось X |

---

## 9. Ссылки

- [HybridMQA](https://github.com/arshafiee/hybridmqa) — CVPR 2025
- [FMQM](https://github.com/yyyykf/FMQM)
- [MEPP2 / MSDM2](https://projet.liris.cnrs.fr/mepp/)
- Streamlit: https://docs.streamlit.io/

---

*Документ актуален для версии с якорной нормировкой ref vs self, загрузкой QV OBJ и интеграцией HybridMQA/FMQM.*

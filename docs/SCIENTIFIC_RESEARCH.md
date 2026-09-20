# Научно-исследовательское описание решения PLER-2.0 Experiment

**Версия документа:** 1.0  
**Дата:** 16.07.2026  
**Назначение:** описание методологии, математических формулировок метрик и принципов работы программного комплекса с точки зрения исследования качества 3D-моделей.

---

## Содержание

1. [Введение](#1-введение)
2. [Постановка задачи](#2-постановка-задачи)
3. [Методология эксперимента](#3-методология-эксперимента)
4. [Архитектура программного комплекса](#4-архитектура-программного-комплекса)
5. [Общий препроцессинг](#5-общий-препроцессинг)
6. [Метрики качества](#6-метрики-качества)
7. [Нормировка и сопоставимость метрик](#7-нормировка-и-сопоставимость-метрик)
8. [Валидация относительно MOS](#8-валидация-относительно-mos)
9. [Ограничения и допущения реализации](#9-ограничения-и-допущения-реализации)
10. [Литература и первоисточники](#10-литература-и-первоисточники)

---

## 1. Введение

**PLER-2.0 Experiment** — программный комплекс для **full-reference (FR)** оценки качества треугольных 3D-мешей в формате OBJ. Для каждой пары *(эталон, искажённая модель)* вычисляется набор объективных метрик; результаты сопоставляются с субъективными оценками **MOS** (Mean Opinion Score) на эталонном датасете QV-сфера.

Комплекс объединяет:

- метрику **PLER** (Projection-based Local Error Representation) и производные индексы;
- классические **геометрические** расстояния между облаками вершин;
- **перцептивные** и **learning-based** метрики (полные или упрощённые реализации);
- **якорную нормировку** для сравнения метрик на единой шкале;
- статистический анализ **корреляций** с MOS.

Документ описывает **именно то, как метрики реализованы в коде** (`metrics_core.py`, `external_metrics.py`, модули PLER), с явным указанием упрощений относительно оригинальных публикаций.

---

## 2. Постановка задачи

### 2.1. Входные данные

- **Эталонная модель** \(M_{\mathrm{ref}}\) — OBJ-меш без искажений (или эталон высокого качества).
- **Искажённая модель** \(M_{\mathrm{dist}}\) — OBJ-меш с артефактами сжатия, децимации, шума и т.п.

### 2.2. Выход

Для каждой пары \((M_{\mathrm{ref}}, M_{\mathrm{dist}})\):

- вектор скalar-метрик \(\mathbf{q} = (q_1, q_2, \ldots, q_K)\);
- опционально — нормированные значения \(\hat{q}_k \in [0, 100]\);
- корреляции \(\rho(q_k, \mathrm{MOS})\) при наличии субъективных оценок.

### 2.3. Дополнительная пара «эталон vs сам себя»

После обработки всех искажённых моделей для каждого уникального эталона добавляется пара:

\[
(M_{\mathrm{ref}}, M_{\mathrm{ref}})
\]

Она служит **якорем идеального качества** при нормировке (см. раздел 7).

---

## 3. Методология эксперимента

### 3.1. Режимы данных

| Режим | Описание |
|-------|----------|
| **QV-сфера (встроенная)** | 13 искажённых `Sphere_1…Sphere_13`, эталон `Sphere_14`, MOS из `data/mos.csv` |
| **QV-сфера (загрузка)** | Пользовательские OBJ с порядком 1…n (1 — наиболее искажённая) |
| **Свои модели** | Генерация контролируемых деградаций (decimation, noise, smoothing, combined) |

### 3.2. Протокол расчёта

1. Формирование списка пар `(reference, distorted)`.
2. Добавление пар ref-vs-self (`pair_utils.append_reference_self_pairs`).
3. Последовательный расчёт выбранных метрик (`compute_selected_metrics`).
4. Сохранение сырых значений в CSV/checkpoint.
5. Обогащение метаданными (число вершин, вес файла, MOS).
6. Якорная нормировка и построение корреляций.

Расчёт **последовательный** (без параллелизма по парам) из-за высокой памятной и вычислительной стоимости PLER на крупных мешах.

---

## 4. Архитектура программного комплекса

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                       │
│  • выбор датасета и метрик                                   │
│  • пайплайн run_pipeline()                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
 pair_utils.py      metrics_core.py      external_metrics.py
 (пары, upload)     PLER, классика       MSDM2, PCQM, …
                           │                     │
                           └──────────┬──────────┘
                                      ▼
                           results_analysis.py
                           (нормировка, MOS, корреляции)
```

**Ключевые модули:**

| Модуль | Роль |
|--------|------|
| `metrics_core.py` | PLER, TSI, AAD, Chamfer, Hausdorff, F-score |
| `external_metrics.py` | Перцептивные метрики (часть — упрощённые) |
| `ml_metrics.py` | Subprocess-вызов HybridMQA / FMQM |
| `metrics/PLER/_Metric_PLER+/` | Оригинальная реализация PLER (ray casting) |
| `results_analysis.py` | Якорная нормировка, PLCC/SROCC/Kendall |

---

## 5. Общий препроцессинг

### 5.1. Загрузка вершин (классические метрики)

Вершины извлекаются из строк `v x y z` OBJ-файла:

\[
\mathbf{V} = \{\mathbf{v}_i \in \mathbb{R}^3\}_{i=1}^{N}
\]

### 5.2. Центрирование (center alignment)

Для классических метрик и AAD облака центрируются относительно **центра эталона**:

\[
\tilde{\mathbf{v}}_i = \mathbf{v}_i - \frac{1}{N_{\mathrm{ref}}}\sum_j \mathbf{v}_j^{\mathrm{(ref)}}
\]

Искажённое облако центрируется тем же вектором (не собственным центром), что обеспечивает инвариантность к трансляции при сохранении относительного положения.

### 5.3. Субсэмплинг

При \(N > N_{\max}\) (по умолчанию \(N_{\max} = 500\,000\) для KDTree, до 200\,000 для trimesh) применяется **детерминированный** случайный отбор:

\[
\mathcal{S} \subset \mathbf{V}, \quad |\mathcal{S}| = N_{\max}, \quad \text{seed} = 0 \text{ или } 1
\]

Это снижает стоимость \(O(N \log N)\) операций KDTree на больших мешах (например, `Sphere_14` ≈ 261k вершин в OBJ, PLER загружает расширенное представление через trimesh).

### 5.4. Масштабирование в миллиметры

Классические расстояния умножаются на константу:

\[
s = 873{,}744 \quad \text{(SCALE\_FACTOR)}
\]

\[
d_{\mathrm{mm}} = s \cdot d_{\mathrm{norm}}
\]

Константа задаёт перевод из нормализованного пространства модели в миллиметры (калибровка проекта PLER-2.0).

### 5.5. Нормализация меша в PLER

В модуле PLER каждый меш:

1. загружается через trimesh;
2. центрируется: \(\mathbf{v} \leftarrow \mathbf{v} - \bar{\mathbf{v}}\);
3. масштабируется в единичную сферу: \(\mathbf{v} \leftarrow \mathbf{v} / \max_i \|\mathbf{v}_i\|\).

Ray casting выполняется из центра координат по направлениям **Fibonacci sphere** (равномерное распределение на \(S^2\)).

---

## 6. Метрики качества

Ниже: **направление** — «↑» (больше = лучше) или «↓» (меньше = лучше).

---

### 6.1. PLER geometry (PLER, dB)

**Класс:** FR-метрика на основе лучевых расстояний до поверхности.  
**Модуль:** `metrics/PLER/_Metric_PLER+/pler_metric.py`  
**Направление:** ↑

#### Принцип

Из центра сферы испускается \(N_{\mathrm{rays}}\) лучей (адаптивно от 1000 до 20000 в зависимости от сложности меша). Для каждого направления \(\mathbf{d}_k\) вычисляется расстояние до пересечения с поверхностью \(t_k^{\mathrm{ref}}\), \(t_k^{\mathrm{dist}}\).

Профиль «радиальной функции»:

\[
L_k = 1 - t_k
\]

**Среднеквадратичная ошибка профилей:**

\[
\mathrm{MSE} = \frac{1}{N_{\mathrm{rays}}} \sum_{k=1}^{N_{\mathrm{rays}}} (L_k^{\mathrm{ref}} - L_k^{\mathrm{dist}})^2
\]

**PLER в децибелах:**

\[
\mathrm{PLER}_{\mathrm{dB}} = 10 \log_{10} \frac{(\max_k L_k^{\mathrm{ref}} - \min_k L_k^{\mathrm{ref}})^2}{\mathrm{MSE}}
\]

При \(\mathrm{MSE} \approx 0\) возвращается 100 dB (идеальное совпадение). Промахи лучей заменяются адаптивным порогом \(1 + 0{,}1 \cdot |V|/1000\).

**Интерпретация:** чем выше PLER (dB), тем ближе искажённая модель к эталону по радиальным профилям.

---

### 6.2. MSE (PLER)

**Направление:** ↓

\[
\mathrm{MSE} = \frac{1}{N_{\mathrm{rays}}} \sum_k (L_k^{\mathrm{ref}} - L_k^{\mathrm{dist}})^2
\]

Без логарифмического масштабирования; используется как вспомогательная величина PLER.

---

### 6.3. TSI (Topology Similarity Index)

**Класс:** топологическое сходство мешей.  
**Модуль:** `topology_analyzer.py`  
**Направление:** ↑

#### Извлекаемые признаки

Для каждого меша вычисляются:

- число вершин \(V\), граней \(F\);
- число уникальных рёбер \(E\) (из триангуляции);
- **характеристика Эйлера:** \(\chi = V - E + F\);
- компоненты связности, род поверхности (упрощённо).

#### Формула сравнения (реализация)

Для ключей `vertex_count`, `face_count`, `connected_components`:

\[
s_j = \begin{cases}
1, & \text{если } x_j^{\mathrm{ref}} = x_j^{\mathrm{dist}} \\
\max\left(0,\; 1 - \dfrac{|x_j^{\mathrm{ref}} - x_j^{\mathrm{dist}}|}{\max(x_j^{\mathrm{ref}}, x_j^{\mathrm{dist}}, 1)}\right), & \text{иначе}
\end{cases}
\]

\[
\mathrm{TSI} = \frac{1}{J}\sum_{j=1}^{J} s_j
\]

**Диапазон:** \([0, 1]\).

---

### 6.4. PLER-2.0 (complex)

**Направление:** ↑

Комбинированный индекс геометрии и топологии:

\[
\mathrm{PLER}_{\mathrm{norm}} = \min\left(\frac{\mathrm{PLER}_{\mathrm{dB}}}{60}, 1\right)
\]

\[
Q_{\mathrm{complex}} = 0{,}7 \cdot \mathrm{PLER}_{\mathrm{norm}} + 0{,}3 \cdot \mathrm{TSI}
\]

Веса 0.7 / 0.3 зафиксированы в `metrics_core.py`.

---

### 6.5. AAD (Average Absolute Deviation)

**Класс:** среднее точечное расстояние после центрирования.  
**Направление:** ↓

Пусть \(\tilde{\mathbf{V}}_{\mathrm{ref}}\), \(\tilde{\mathbf{V}}_{\mathrm{dist}}\) — центрированные наборы вершин. Для каждой вершины искажённого облака находится ближайшая вершина эталона (KDTree):

\[
\mathrm{AAD} = \frac{1}{N_{\mathrm{dist}}} \sum_{i=1}^{N_{\mathrm{dist}}} \min_{j} \|\tilde{\mathbf{v}}_i^{\mathrm{dist}} - \tilde{\mathbf{v}}_j^{\mathrm{ref}}\|
\]

**Примечание:** в коде это называется «MCM»-подобная метрика; единицы — нормализованное пространство (без SCALE_FACTOR).

---

### 6.6. Классические метрики расстояния

Все вычисляются в `compute_classic_metrics()` после center alignment и субсэмплинга.

#### 6.6.1. Mean (mm), RMS (mm), Max (mm), P95 (mm), Std (mm)

Пусть \(d_i\) — расстояния от \(i\)-й вершины искажённого облака до ближайшей вершины эталона (dist → ref), в мм:

\[
\mathrm{Mean} = \frac{1}{N}\sum_i d_i, \quad
\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_i d_i^2}
\]

\[
\mathrm{Max} = \max_i |d_i|, \quad
\mathrm{P95} = \mathrm{percentile}_{95}(|d_i|), \quad
\mathrm{Std} = \mathrm{std}(d_i)
\]

**Направление:** ↓ для всех.

#### 6.6.2. Chamfer (mm)

Симметричное среднее расстояние (Chamfer):

\[
\mathrm{CD} = \frac{1}{2}\left(
\frac{1}{N_{\mathrm{dist}}}\sum_{i} \|\mathbf{p}_i - \mathcal{N}_{\mathrm{ref}}(\mathbf{p}_i)\|
+
\frac{1}{N_{\mathrm{ref}}}\sum_{j} \|\mathbf{q}_j - \mathcal{N}_{\mathrm{dist}}(\mathbf{q}_j)\|
\right) \cdot s
\]

где \(\mathcal{N}(\cdot)\) — ближайший сосед, \(s\) — SCALE_FACTOR.

**Направление:** ↓

#### 6.6.3. Hausdorff (mm)

\[
H(P, Q) = \max\left(
\max_{p \in P} \min_{q \in Q} \|p - q\|,\;
\max_{q \in Q} \min_{p \in P} \|p - q\|
\right) \cdot s
\]

**Направление:** ↓

#### 6.6.4. F-score @ τ

Для порога \(\tau \in \{0{,}01, 0{,}02, 0{,}05, 0{,}1\}\) (в нормализованных единицах):

\[
\mathrm{Precision}(\tau) = \frac{1}{|P|}\sum_{p \in P} \mathbb{1}[\min_{q \in Q}\|p-q\| \le \tau]
\]

\[
\mathrm{Recall}(\tau) = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}[\min_{p \in P}\|p-q\| \le \tau]
\]

\[
F(\tau) = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
\]

**Направление:** ↑

---

### 6.7. MSDM2 (упрощённая реализация)

**Оригинал:** Nehmé et al., MSDM2 (MEPP2).  
**В решении:** упрощённый прокси в `external_metrics.compute_msdm2`.  
**Направление:** ↑

1. С поверхности каждого меша сэмплируется \(N_s = 10\,000\) точек (`trimesh.sample.sample_surface`).
2. Для каждой точки искажённого меша \( \mathbf{p} \) находится расстояние до ближайшей точки эталона \(d(\mathbf{p})\).
3. Ширина ядра: \(\sigma = 0{,}05 \cdot \mathrm{mean}(\mathrm{std}(\mathbf{X}_{\mathrm{ref}})) + \varepsilon\).
4. Вес: \(w(\mathbf{p}) = \exp(-d(\mathbf{p})^2 / (2\sigma^2))\).

\[
\mathrm{MSDM2}_{\mathrm{proxy}} = \mathrm{clip}\left(\frac{1}{N_s}\sum w(\mathbf{p}),\; 0,\; 1\right)
\]

**Отличие от оригинала:** нет многошкального MSDM по кривизне и цвету; используется гауссово ядро по геометрическому расстоянию сэмплов.

---

### 6.8. PCQM (упрощённая реализация)

**Оригинал:** MPEG Point Cloud Quality Metric.  
**Направление:** ↑

1. Загрузка вершин (до 200k), center alignment.
2. Расстояния dist → ref: \(d_i\).
3. Нормировка: \(\hat{d}_i = d_i / \mathrm{P95}(d)\).
4. PSNR-подобная величина:

\[
\mathrm{PSNR}_{\mathrm{proxy}} = -10 \log_{10}\left(\frac{1}{N}\sum_i \hat{d}_i^2 + \varepsilon\right)
\]

\[
\mathrm{PCQM}_{\mathrm{proxy}} = \mathrm{clip}(\mathrm{PSNR}_{\mathrm{proxy}} / 60,\; 0,\; 1)
\]

---

### 6.9. 3D-PSSIM (упрощённая реализация)

**Направление:** ↑

1. Загрузка мешей в Open3D.
2. Для 6 ортогональных направлений взгляда \(\mathbf{v} \in \{\pm e_x, \pm e_y, \pm e_z\}\) строится **ортографическая depth-карта** \(128 \times 128\) (z-buffer по вершинам).
3. Между картами эталона и искажения вычисляется **упрощённый SSIM**:

\[
\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\]

\(C_1 = (0{,}01)^2\), \(C_2 = (0{,}03)^2\).

\[
\mathrm{3D\text{-}PSSIM}_{\mathrm{proxy}} = \frac{1}{6}\sum_{\mathbf{v}} \mathrm{SSIM}(I_{\mathrm{ref}}^{\mathbf{v}}, I_{\mathrm{dist}}^{\mathbf{v}})
\]

---

### 6.10. GeodesicPSIM (упрощённая реализация)

**Оригинал:** C++ GeodesicPSIM (геодезические расстояния на поверхности).  
**Направление:** ↑

Прокси по **нормалям вершин** (до 20k пар):

\[
\cos_i = \frac{\mathbf{n}_i^{\mathrm{ref}} \cdot \mathbf{n}_i^{\mathrm{dist}}}{\|\mathbf{n}_i^{\mathrm{ref}}\|\,\|\mathbf{n}_i^{\mathrm{dist}}\|}
\]

\[
\mathrm{GeodesicPSIM}_{\mathrm{proxy}} = \mathrm{clip}\left(\frac{1}{N}\sum_i \frac{\cos_i + 1}{2},\; 0,\; 1\right)
\]

**Примечание:** сопоставление вершин — по индексу (первые \(N\) вершин), не по параметризации поверхности.

---

### 6.11. PointPCA+

**Направление:** ↑

1. Для каждого из 500 опорных точек берётся \(k=30\) ближайших соседей.
2. PCA патча → спектр собственных значений \(\lambda_1 \le \lambda_2 \le \lambda_3\).
3. Усреднённый спектр \(\bar{\boldsymbol{\lambda}}^{\mathrm{ref}}\), \(\bar{\boldsymbol{\lambda}}^{\mathrm{dist}}\).

\[
\mathrm{err} = \frac{\|\bar{\boldsymbol{\lambda}}^{\mathrm{ref}} - \bar{\boldsymbol{\lambda}}^{\mathrm{dist}}\|}{\|\bar{\boldsymbol{\lambda}}^{\mathrm{ref}}\| + \varepsilon}
\]

\[
\mathrm{PointPCA+} = \mathrm{clip}(1 - \mathrm{err},\; 0,\; 1)
\]

---

### 6.12. PQI (Point Quality Index, упрощённый)

**Направление:** ↑

\[
\mathrm{PQI} = 0{,}6 \cdot \mathrm{PCQM}_{\mathrm{proxy}} + 0{,}4 \cdot \mathrm{MSDM2}_{\mathrm{proxy}}
\]

---

### 6.13. CMDM

**Оригинал:** Color Mesh Distortion Metric (Nehmé et al., vertex colors).  
**Направление:** ↑

При наличии vertex colors:

\[
e_i = 0{,}5 \cdot \frac{\|\mathbf{v}_i^{\mathrm{ref}} - \mathbf{v}_i^{\mathrm{dist}}\|}{\max_j \|\mathbf{v}_j^{\mathrm{ref}} - \mathbf{v}_j^{\mathrm{dist}}\|}
+ 0{,}5 \cdot \|\mathbf{c}_i^{\mathrm{ref}} - \mathbf{c}_i^{\mathrm{dist}}\|
\]

\[
\mathrm{CMDM} = \mathrm{clip}(1 - \mathrm{mean}(e_i),\; 0,\; 1)
\]

Без vertex colors → **NaN**.

---

### 6.14. HybridMQA

**Оригинал:** arshafiee et al., CVPR 2025 — гибридная FR-метрика для textured/colored mesh.  
**В решении:** полный inference через subprocess (`ml_metrics.py`) при установке:

```bash
python scripts/setup_ml_metrics.py
```

Требует checkpoint `ckpt_TMQA_adapted.pth` и репозиторий `third_party/hybridmqa`. Без установки → **NaN**.

**Направление:** ↑ (выше = лучше perceptual quality).

---

### 6.15. FMQM

**Оригинал:** Full-reference Mesh Quality Metric (текстурированные меши).  
**В решении:** subprocess к `fmqm_single_mesh_eval.py` при наличии OBJ + PNG/JPG текстур.  
**Направление:** ↑

---

## 7. Нормировка и сопоставимость метрик

### 7.1. Якорная нормировка (основная в UI)

Для каждой метрики \(q\) в группе с одним эталоном:

- \(q_{\mathrm{ref}}^{\mathrm{self}}\) — значение на паре ref-vs-self;
- \(q_{\mathrm{worst}}\) — худшее среди искажённых:
  - для метрик ↑: \(q_{\mathrm{worst}} = \min q\);
  - для метрик ↓: \(q_{\mathrm{worst}} = \max q\).

**Метрики «чем больше, тем лучше» (↑):**

\[
\hat{q} = 100 \cdot \frac{q - q_{\mathrm{worst}}}{q_{\mathrm{ref}}^{\mathrm{self}} - q_{\mathrm{worst}}}
\]

**Метрики «чем меньше, тем лучше» (↓):**

\[
\hat{q} = 100 \cdot \frac{q_{\mathrm{worst}} - q}{q_{\mathrm{worst}} - q_{\mathrm{ref}}^{\mathrm{self}}}
\]

Результат обрезается в \([0, 100]\). Для ref-vs-self всегда \(\hat{q} = 100\).

**Смысл:** разные метрики приводятся к **общей шкале относительного качества** внутри эксперимента, что позволяет сравнивать их на одном графике.

### 7.2. Нормировка MOS

\[
\widehat{\mathrm{MOS}} = 100 \cdot \frac{\mathrm{MOS}}{\mathrm{MOS}_{\mathrm{ref}}^{\mathrm{self}}}
\]

(или max MOS в группе, если ref-self MOS отсутствует).

### 7.3. Теоретическая нормировка (альтернатива в коде)

По фиксированным границам \([q_{\min}, q_{\max}]\) из `METRIC_THEORETICAL_BOUNDS` — доступна через `normalization="theoretical"`, в UI по умолчанию не используется.

---

## 8. Валидация относительно MOS

### 8.1. Субъективные оценки

MOS загружается из CSV (`data/mos.csv` или пользовательский файл). Формат: transposed (`model_name;Sphere_3;…` / `mos;1.25;…`) или long (`model_name, mos`).

### 8.2. Корреляционный анализ

Для каждой метрики \(q_k\) (строка ref-vs-self **исключается**):

| Коэффициент | Определение |
|-------------|-------------|
| **PLCC** | Pearson linear correlation |
| **SROCC** | Spearman rank correlation |
| **Kendall τ** | Kendall rank correlation |

Вычисляются также p-value для проверки статистической значимости (\(\alpha = 0{,}05\)).

**Интерпретация:** высокие PLCC/SROCC при \(p < 0{,}05\) указывают на согласованность объективной метрики с субъективным восприятием на данном датасете.

---

## 9. Ограничения и допущения реализации

| Аспект | Ограничение |
|--------|-------------|
| **PLER** | Высокая стоимость на больших мешах; trimesh может разворачивать OBJ в сотни тысяч вершин |
| **Субсэмплинг** | Классические метрики стохастически аппроксимируют полное облако |
| **MSDM2, PCQM, 3D-PSSIM, GeodesicPSIM** | Упрощённые прокси; **не эквивалентны** эталонным реализациям MEPP2/MPEG |
| **PointPCA+** | Сопоставление патчей без явной регистрации поверхностей |
| **HybridMQA / FMQM** | Требуют отдельной установки, текстур и GPU/CPU PyTorch |
| **Регистрация** | Только center alignment; нет ICP / rigid alignment |
| **Streamlit UI** | Длительные вычисления блокируют один поток; прогресс обновляется между этапами пары |

### Эмпирическая оценка времени (QV-сфера, 23 метрики)

По бенчмарку на 14 парах (13 Sphere + ref-self):

- **~17–20 мин** на полный прогон (Intel/Windows, CPU);
- **~74 с/пара** в среднем;
- наиболее затратны пары с крупным `Sphere_13` и ref-vs-self (PLER ray casting на полном эталоне).

---

## 10. Литература и первоисточники

| Метрика | Ссылка |
|---------|--------|
| PLER | Модуль `metrics/PLER/_Metric_PLER+/`, проект PLER-2.0 |
| MSDM2, CMDM | Nehmé et al., MEPP2 — https://projet.liris.cnrs.fr/mepp/ |
| PCQM | MPEG PCC — https://github.com/MPEGGroup/mpeg-pcc-rendering |
| HybridMQA | arshafiee et al., CVPR 2025 — https://github.com/arshafiee/hybridmqa |
| FMQM | https://github.com/yyyykf/FMQM |
| GeodesicPSIM | https://github.com/Qi-Yangsjtu/GeodesicPSIM |
| Chamfer / Hausdorff / F-score | Классические метрики облаков точек |
| SSIM | Wang et al., 2004 |

---

## Приложение A. Сводная таблица метрик

| Метрика | Направление | Диапазон (тип.) | Реализация |
|---------|-------------|-----------------|------------|
| PLER geometry | ↑ | \([0, 100]\) dB | Полная (PLER+) |
| PLER-2.0 (complex) | ↑ | \([0, 1]\) | Полная + TSI |
| TSI | ↑ | \([0, 1]\) | Полная (topology) |
| MSE | ↓ | \([0, 1]\) | PLER |
| AAD | ↓ | norm. units | Полная |
| Chamfer, Hausdorff, Mean, Max, RMS, P95, Std | ↓ | mm | Полная |
| F-score @ τ | ↑ | \([0, 1]\) | Полная |
| MSDM2 | ↑ | \([0, 1]\) | **Упрощённая** |
| PCQM | ↑ | \([0, 1]\) | **Упрощённая** |
| 3D-PSSIM | ↑ | \([0, 1]\) | **Упрощённая** |
| GeodesicPSIM | ↑ | \([0, 1]\) | **Упрощённая** |
| PointPCA+ | ↑ | \([0, 1]\) | **Упрощённая** |
| PQI | ↑ | \([0, 1]\) | **Комбинация** |
| CMDM | ↑ | \([0, 1]\) | Упрощённая (нужен цвет) |
| HybridMQA | ↑ | model-specific | **Полная*** |
| FMQM | ↑ | model-specific | **Полная*** |

\* при установке `scripts/setup_ml_metrics.py`

---

## Приложение B. Схема вычисления одной пары

```
(ref.obj, dist.obj)
        │
        ├─► PLER ray casting ──► PLER dB, MSE
        ├─► TopologyAnalyzer ──► TSI, PLER-2.0 complex
        ├─► KDTree vertices ──► AAD, Chamfer, Hausdorff, F-score, …
        └─► external_metrics ──► MSDM2, PCQM, 3D-PSSIM, …
                │
                ▼
         raw metrics vector q
                │
                ├─► append ref-vs-self row
                ├─► anchor normalization → q̂ ∈ [0,100]
                └─► correlation with MOS
```

---

*Документ отражает состояние кодовой базы PLER-2.0 Experiment на 16.07.2026. При изменении формул в коде следует обновлять соответствующие разделы.*

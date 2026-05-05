# Анализ сложности ребусов

## 1. Цель

Цель проекта - построить модель, которая предсказывает сложность ребуса по доступным данным: текстовому ответу, текстовому описанию и изображению ребуса.

Основная метрика сравнения моделей - `MAE` на кросс-валидации: чем ниже ошибка, тем точнее модель предсказывает числовую сложность. Дополнительно считаются `RMSE` и `R2`.

## 2. Допустимые объекты

В проекте используются только объекты, для которых есть строка в исходном датасете `data.csv`.

Для каждого объекта доступны:

- `answer` - правильный ответ или варианты ответа через `|`;
- `description` - текстовое описание логики ребуса;
- `img_url` - путь к изображению;
- целевой столбец `difficulty`.

В таблице также есть поведенческие статистики прохождения: `started`, `solved`, `users_with_hints`, `total_hints`, `solve_rate`, `hint_usage`, `avg_hints`. Они считаются служебными leakage-колонками, потому что `difficulty` вычисляется напрямую по формуле от этих величин. Эти столбцы не используются как признаки модели.

Формула расчета `difficulty` фиксируется в `DIFFICULTY_FORMULA` в `notebooks/00_data_eda_preprocessing.ipynb`:

```text
если started == 0:
    difficulty = 0
иначе:
    difficulty = 0.6 * (1 - solved / started)
               + 0.25 * (users_with_hints / started)
               + 0.15 * (total_hints / started)
```

Все участвующие в формуле столбцы остаются в `LEAKAGE_COLUMNS` и запрещены как признаки модели.

## 3. Этапы проекта

### 3.1. Постановка задачи

Задача решается как регрессия: по признакам ребуса нужно предсказать непрерывное значение `difficulty`.

Проверяются несколько групп признаков:

- только признаки из `answer`;
- признаки из `answer` и `description`;
- NLP-признаки из `answer` и `description`;
- визуальные признаки EfficientNet;
- объединение лучшего текстового подхода с EfficientNet.

### 3.2. Данные

Исходный датасет:

- `data.csv` - базовая таблица с ответами, описаниями, ссылками на изображения, статистиками прохождения и `difficulty`.

Промежуточные датасеты:

- `data_with_efficientnet_features.csv` - датасет с признаками EfficientNet-B0.

Основной датасет для финальных визуальных экспериментов - `data_with_efficientnet_features.csv`, потому что он содержит исходные поля и колонки `effnet_feat_*`.

### 3.3. Предобработка

Проект сохраняет постепенное изменение датасета:

1. `data.csv` используется для текстовых baseline-экспериментов.
2. `extract_efficientnet_features.py` извлекает EfficientNet-B0 эмбеддинги изображений и формирует `data_with_efficientnet_features.csv`.
3. Отдельный ноутбук проверяет обучение только на `effnet_feat_*`, без `answer` и `description` в матрице признаков.
4. Финальный ноутбук выбирает лучший текстовый подход и объединяет его с EfficientNet-признаками.

Запрещенные leakage-колонки не используются в обучении:

- `started`;
- `solved`;
- `users_with_hints`;
- `total_hints`;
- `solve_rate`;
- `hint_usage`;
- `avg_hints`.

Текстовая предобработка включает:

- приведение ответа к первому варианту до `|`;
- статистики длины, букв, гласных, редких букв, уникальных символов;
- для NLP-подхода: лемматизацию, TF-IDF по описанию и char n-gram TF-IDF по ответу.

Визуальная предобработка включает:

- загрузку изображения;
- применение предобученной EfficientNet-B0 без классификационной головы;
- сохранение 1280 признаков `effnet_feat_*`;
- PCA внутри `Pipeline`/`ColumnTransformer`, чтобы не было утечки данных между fold'ами.

## 4. Структура файлов

Ноутбуки разделены по этапам:

- `notebooks/00_data_eda_preprocessing.ipynb` - данные, предобработка и EDA;
- `notebooks/01_answer_features_training.ipynb` - baseline только по `answer`;
- `notebooks/02_answer_description_features_training.ipynb` - статистические признаки `answer + description`;
- `notebooks/03_nlp_text_features_training.ipynb` - NLP-подход;
- `notebooks/04_efficientnet_features_training.ipynb` - обучение только по EfficientNet-признакам;
- `notebooks/05_best_text_plus_efficientnet_training.ipynb` - финальное объединение лучшего текстового подхода с EfficientNet;
- `notebooks/06_approaches_metrics_report.ipynb` - сводные метрики и графики по всем подходам.
- `notebooks/07_hyperparameter_tuning_interpretation_saving.ipynb` - подбор гиперпараметров, интерпретация и сохранение модели;
- `notebooks/08_hypothesis_testing.ipynb` - проверка гипотез.

Скрипты:

- `extract_efficientnet_features.py` - извлечение EfficientNet-B0 признаков;
- `run_experiments.py` - единый запуск сравнения подходов через переиспользуемые `.py` модули;
- `src/puzzle_difficulty/` - общая логика формулы, feature engineering, моделей и метрик;
- `requirements.txt` - зависимости проекта.

Изображения:

- `images/daily-rebuses/` - рабочая папка с изображениями ребусов;
- `images/README.md` - правила сопоставления `img_url` с локальными файлами.

## 5. Оценка качества

Для выбора моделей используется 5-fold cross-validation.

Основные метрики:

- `MAE` - средняя абсолютная ошибка, основная метрика;
- `RMSE` - сильнее штрафует крупные ошибки;
- `R2` - показывает долю объясненной дисперсии.

`Holdout` используется как дополнительная финальная проверка выбранной модели на отложенной части данных, но не как основной критерий выбора.

## 6. Соответствие проектному заданию

| Пункт ТЗ | Где выполнено |
| --- | --- |
| 1. Цель: полный цикл данные -> модель -> оценка -> интерпретация -> сохранение | `README.md`, `00_data_eda_preprocessing.ipynb`, `05_best_text_plus_efficientnet_training.ipynb`, `06_approaches_metrics_report.ipynb`, `07_hyperparameter_tuning_interpretation_saving.ipynb` |
| 2. Допустимая область | Область: IT / цифровые продукты и социальные данные. Используются ребусы, тексты и изображения; поведенческие статистики только объясняют происхождение target и не используются как признаки |
| 3.1 Постановка задачи | `README.md`, раздел 3.1; задача регрессии по `difficulty` |
| 3.2 Данные: реальный датасет и описание признаков | `README.md`, раздел 3.2; `00_data_eda_preprocessing.ipynb`, блок "Описание признаков" |
| 3.3 Предобработка: пропуски | `00_data_eda_preprocessing.ipynb`, блок "Пропуски и типы данных" |
| 3.3 Предобработка: кодирование категорий | В проекте нет обычных категориальных признаков для one-hot encoding; `answer` и `description` кодируются через статистики, лемматизацию и TF-IDF. Описано в `00_data_eda_preprocessing.ipynb` |
| 3.3 Предобработка: нормализация | `StandardScaler` внутри `Pipeline`/`ColumnTransformer` в ноутбуках `01`, `02`, `04`, `05`, `06`, `07` |
| 3.3 Feature engineering | Текстовые статистики в `01` и `02`, NLP-признаки в `03`, EfficientNet-признаки в `extract_efficientnet_features.py` и `04`; переиспользуемые функции вынесены в `src/puzzle_difficulty/features.py` |
| 3.4 EDA: минимум 5 графиков | `00_data_eda_preprocessing.ipynb`: распределение `difficulty`, длина `answer`, `answer_len` vs `difficulty`, `description_len` vs `difficulty`, `variant_count` vs `difficulty`, корреляционная матрица разрешенных признаков |
| 3.4 EDA: распределения | `00_data_eda_preprocessing.ipynb`, график распределения `difficulty` и анализ числовых признаков |
| 3.4 EDA: корреляции | `00_data_eda_preprocessing.ipynb`, корреляционная матрица |
| 3.4 EDA: минимум 3 вывода | `00_data_eda_preprocessing.ipynb`, блок "Выводы EDA" |
| 3.5 Baseline модель | `01_answer_features_training.ipynb`: `DummyRegressor`, `LinearRegression`, `Ridge` |
| 3.5 Классические ML-модели | `01`, `02`, `03`, `04`, `05`: Ridge, Lasso, ElasticNet, SVR, KNN |
| 3.5 Ансамблевые модели | `01`, `02`, `03`, `04`, `05`: RandomForest, ExtraTrees, GradientBoosting |
| 3.5 Опционально глубокое обучение | `extract_efficientnet_features.py`: EfficientNet-B0 как предобученный extractor визуальных признаков |
| 3.6 Оценка качества Regression: RMSE, MAE, R2 | `06_approaches_metrics_report.ipynb`, также в обучающих ноутбуках |
| 3.6 Сравнительная таблица моделей | `06_approaches_metrics_report.ipynb`, блоки "Кросс-валидация", "Таблица всех моделей" |
| 3.7 Подбор гиперпараметров | `07_hyperparameter_tuning_interpretation_saving.ipynb`, `GridSearchCV` для финального pipeline |
| 3.8 Интерпретация: Feature importance | `07_hyperparameter_tuning_interpretation_saving.ipynb`, permutation importance |
| 3.8 Объяснение решений модели | `07_hyperparameter_tuning_interpretation_saving.ipynb`, блок "Объяснение решений модели" и таблица ошибок |
| 3.9 Проверка гипотез | `08_hypothesis_testing.ipynb` |
| 4. Архитектура Data -> Preprocessing -> Model -> Evaluation -> Сохранение модели | Реализовано по файлам `00` -> `01-05` -> `06` -> `07`; сохранение через `joblib` в `07`; переиспользуемая версия пайплайна есть в `run_experiments.py` и `src/puzzle_difficulty/` |
| 5. GitHub репозиторий | Код, ноутбуки, `README.md`, `requirements.txt` подготовлены для загрузки в GitHub |
| 5. Презентация | Пока не создана отдельным файлом; материалы и таблицы для нее собраны в `README.md` и `06_approaches_metrics_report.ipynb` |

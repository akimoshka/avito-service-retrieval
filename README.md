# Avito Service Retrieval

Решение тестового задания по генерации кандидатов для поиска услуг Avito.

## Задача

Для каждого поискового запроса из `benchmark_queries.parquet` необходимо выбрать до 50 наиболее релевантных объявлений из `benchmark_items.parquet`.

Итоговая метрика - средний `Recall@50` по запросам.

Поскольку порядок объявлений внутри Top-50 не влияет на метрику, задача рассматривается как candidate generation, а не как задача финального ranking.

Данные не добавляются в Git-репозиторий и должны быть расположены локально в `data/raw/`.

---

## Данные и признаки

Использовались три набора данных:
- `train.parquet` - исторические положительные взаимодействия query–item
- `benchmark_queries.parquet` - 2 452 поисковых запроса
- `benchmark_items.parquet` - 189 212 объявлений

`train.parquet` содержит 497 673 положительных взаимодействия.

Для lexical retrieval использовались следующие признаки объявления:
- `item_title_raw`
- `item_infm_params_text`
- `item_description_raw`

Они объединяются в единое текстовое представление объявления:

```text
item_title_raw + item_infm_params_text + item_description_raw
```

Для запроса используется `search_query`.

Дополнительно используется `search_location_id` / `item_location_id` как географический сигнал релевантности. Также `train.parquet` используется как дополнительный historical retrieval канал для запросов, которые ранее встречались в истории.

---

## Анализ данных

### Пересечение запросов

В train содержится 74 529 уникальных текстов запросов. Около **37% benchmark-запросов** имеют точное текстовое совпадение с запросами из train. Это показало, что исторические взаимодействия можно использовать как дополнительный retrieval-сигнал. При этом большая часть benchmark-запросов не имеет exact match, поэтому historical retrieval не может быть единственным способом генерации кандидатов.

### Количество релевантных объявлений

Для большинства запросов количество положительных объявлений невелико:
- 60.58% запросов имеют одно релевантное объявление
- 74.81% - не более двух
- 87.54% - не более пяти

Поэтому для метрики особенно важно обеспечить высокий recall candidate generator и не потерять релевантное объявление до формирования Top-50.

### Категория

Практически все положительные взаимодействия относятся к `item_category_id = 114`. При этом около 99% benchmark corpus также принадлежит этой категории. Поэтому жёсткая фильтрация по категории практически не уменьшает пространство поиска и в финальном решении не используется.

### Локация

Для примерно **83.1%** положительных взаимодействий выполняется:

```text
search_location_id == item_location_id
```

Таким образом, географическое совпадение является сильным сигналом релевантности. При этом location не используется как жёсткий фильтр: около 17% положительных взаимодействий происходят между различными локациями. Вместо фильтрации используется мягкий location boost.

---

## Validation

Для локальной оценки был сформирован validation split по уникальным текстам `search_query`.

Случайно выбирались 2 452 уникальных поисковых запроса, после чего все взаимодействия с этими текстами полностью исключались из retrieval train. Таким образом, validation моделирует строгий сценарий работы с unseen queries.

Для каждого query context формируется множество релевантных `item_id`, после чего качество оценивается с помощью среднего `Recall@50`. Validation corpus состоит из benchmark items и положительных объявлений validation. Все relevant items присутствуют в validation corpus.

Такой split является более строгим, чем реальный benchmark: часть benchmark-запросов встречается в train. Поэтому локальный Recall использовался прежде всего для сравнения retrieval-подходов, а не как точная оценка leaderboard score.

---

## Lexical retrieval

В качестве основного candidate generator используется **TF-IDF (Term Frequency - Inverse Document Frequency)**.

TF-IDF преобразует тексты в разреженные числовые векторы, увеличивая вес терминов, характерных для конкретного документа, и уменьшая влияние слов, часто встречающихся во всей коллекции. Релевантность объявления запросу определяется через **cosine similarity** между TF-IDF-вектором `search_query` и TF-IDF-вектором объявления.

Используются отдельные слова и пары соседних слов (`ngram_range=(1, 2)`), что позволяет учитывать характерные для поиска услуг словосочетания, например `ремонт холодильника` или `установка кондиционера`.

### Эксперименты с текстовым представлением

| Эксперимент | Представление объявления | Local Recall@50 |
|---|---|---:|
| B1 | title | 0.1880 |
| B2 | title + params | 0.1882 |
| B3 | title + params + description | **0.2328** |
| B4 | title + description | 0.2131 |
| B5 | Character TF-IDF | 0.2240 |

Лучшим чисто текстовым вариантом оказался:

```text
item_title_raw + item_infm_params_text + item_description_raw
```

Для поискового запроса используется `search_query`.

### TF-IDF параметры

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
)
```

Retrieval выполняется батчами с использованием sparse matrix multiplication, что позволяет не строить полную dense query-item матрицу.

---

## Location-aware retrieval

EDA показал сильную зависимость релевантности от географического положения. Поэтому к lexical similarity добавляется мягкий bonus за совпадение локации:

```text
score(q, i) = cosine(TFIDF(q), TFIDF(i)) + α · I(location_q = location_i)
```

где:
- `cosine(TFIDF(q), TFIDF(i))` - lexical similarity запроса и объявления
- `I(location_q = location_i)` - индикатор совпадения локации
- `α` - коэффициент location boost

На локальной validation были проверены:

| Location boost α | Local Recall@50 |
|---:|---:|
| 0.05 | 0.4183 |
| 0.10 | 0.5087 |
| **0.20** | **0.5392** |

Лучшим среди проверенных вариантов оказался `α = 0.20`.

По сравнению с чистым lexical baseline:

```text
TF-IDF                    0.2328
TF-IDF + location         0.5392
```

Location оказался одним из наиболее сильных признаков задачи.

---

## Historical retrieval

EDA показал, что часть benchmark-запросов уже встречалась в `train.parquet`. Поэтому в финальный pipeline был добавлен дополнительный historical retrieval channel.

Для benchmark-запроса выполняется поиск совпадающего `search_query` в train. Из найденных взаимодействий используются только `item_id`, которые присутствуют в `benchmark_items.parquet`.

При формировании historical candidates сначала рассматриваются объявления из той же `search_location_id`, затем объявления для того же текста запроса из других локаций.

Количество historical candidates ограничено:

```text
HISTORICAL_K = 15
```

Historical candidates объединяются с lexical candidates без дубликатов.

На benchmark добавление historical retrieval улучшило Recall@50:

```text
TF-IDF + location                 0.503211
+ historical retrieval            0.505049
```

Historical retrieval дал небольшой положительный прирост и поэтому был сохранён в финальном решении.

---

## Анализ ошибок и эксперименты

В процессе разработки анализировались не только итоговые значения Recall@50, но и причины потери релевантных объявлений.

### 1. Недостаточно текстовых признаков объявления

Использование только `item_title_raw` давало низкий Recall@50.

Добавление `item_infm_params_text` практически не изменило результат, однако добавление `item_description_raw` заметно увеличило полноту retrieval.

Поэтому в финальном варианте используются:

```text
title + params + description
```

### 2. Character TF-IDF не улучшил Word TF-IDF

Character TF-IDF рассматривался как способ лучше обрабатывать опечатки и морфологические вариации.

Однако локальный результат составил `0.2240` против `0.2328` у Word TF-IDF с полным текстом объявления.

Поэтому Character TF-IDF не использовался в финальном pipeline.

### 3. Игнорирование локации приводило к большому числу нерелевантных кандидатов

EDA показал совпадение query/item location примерно в 83.1% положительных взаимодействий.

Добавление location boost повысило local Recall@50 с `0.2328` до `0.5392`.

При этом location не используется как hard filter, поскольку это привело бы к потере примерно 17% положительных взаимодействий с несовпадающими локациями.

### 4. Добавление query params ухудшило качество

Был протестирован вариант, в котором запрос представлялся как:

```text
search_query + search_infm_params_text
```

На benchmark качество снизилось:

```text
search_query                         0.505049
search_query + search params         0.495976
```

Поэтому в финальной версии для текстового представления запроса используется только `search_query`.

### 5. Слишком маленький lexical candidate pool

Изначально location boost применялся только к Top-500 объявлениям по lexical similarity.

Это оказалось существенным ограничением: объявление из правильной локации не могло попасть в Top-50, если оно находилось ниже первых 500 lexical candidates.

Поэтому размер промежуточного candidate pool был увеличен.

| Candidate pool | Benchmark Recall@50 |
|---:|---:|
| 500 | 0.505049 |
| 5 000 | 0.641162 |
| **20 000** | **0.649150** |

Это стало самым значительным улучшением финального решения.

Результат показывает, что для данной задачи важно сначала обеспечить достаточно широкий lexical recall, а уже затем использовать location signal для отбора итоговых кандидатов.

### 6. Проверка коэффициента location boost

Дополнительно был протестирован меньший коэффициент:

```text
α = 0.15 -> Benchmark Recall@50 = 0.503293
α = 0.20 -> результат выше
```

Поэтому в финальной конфигурации сохранён `LOCATION_BOOST = 0.20`.

---

## Итоговые эксперименты

Основные итерации решения:

| Вариант | Recall@50 | Тип оценки |
|---|---:|---|
| Word TF-IDF: title | 0.1880 | local validation |
| Word TF-IDF: title + params | 0.1882 | local validation |
| Word TF-IDF: title + params + description | 0.2328 | local validation |
| Character TF-IDF | 0.2240 | local validation |
| TF-IDF + location (`α=0.20`) | 0.5392 | local validation |
| Initial benchmark, candidate pool = 500 | 0.503211 | benchmark |
| + historical retrieval | 0.505049 | benchmark |
| + query params | 0.495976 | benchmark |
| location `α=0.15` | 0.503293 | benchmark |
| candidate pool = 5 000 | 0.641162 | benchmark |
| **candidate pool = 20 000** | **0.649150** | **benchmark** |

---

## Финальный подход

Финальный candidate generator состоит из следующих этапов:

1. Объединение `item_title_raw`, `item_infm_params_text` и `item_description_raw`
2. Построение Word TF-IDF с unigram + bigram признаками
3. Вычисление cosine similarity между `search_query` и benchmark items
4. Выбор расширенного набора до **20 000 lexical candidates**
5. Добавление `0.20` к score объявления при совпадении `search_location_id` и `item_location_id`
6. Выбор итоговых Top-50 lexical/location candidates
7. Добавление до 15 historical candidates для запросов, встречавшихся в train, с приоритетом совпадающей локации
8. Удаление дубликатов и сохранение Top-50
9. Для редких запросов с недостаточным количеством кандидатов используется fallback: сначала объявления из той же локации, затем объявления из общего benchmark corpus

Финальные параметры:

```text
TOP_K = 50
CANDIDATE_K = 20000
LOCATION_BOOST = 0.20
HISTORICAL_K = 15
```

**Финальный benchmark Recall@50: `0.649150`**

---

## Запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Исходные файлы должны находиться в:

```text
data/raw/train.parquet
data/raw/benchmark_queries.parquet
data/raw/benchmark_items.parquet
```

### 3. Генерация submission

Из корня проекта:

```bash
python scripts/generate_submission.py
```

Скрипт:

- загружает train, benchmark queries и benchmark items
- строит TF-IDF представление
- выполняет lexical retrieval
- применяет location boost
- добавляет historical candidates
- выбирает Top-50
- выполняет проверки результата
- сохраняет submission

Результат:

```text
outputs/answer.csv
```

Формат:

```csv
query_id,answer
70DfDUpwjxB4lzFd,6042fecdd7753112 f4e8903d7689e222 603623b4bd9e8f6c ...
JTrdTaZJvSiLPkXj,cbeccbecb1fb8d86 5e62c7a98f91a124 ...
```

Каждая строка соответствует одному `query_id`. Колонка `answer` содержит до 50 различных `item_id`, записанных через один пробел. `query_id` и `item_id` сохраняются как строки без изменения регистра.

---

## Проверки submission

Перед сохранением результата автоматически проверяется:

- наличие ровно двух колонок: `query_id` и `answer`
- наличие всех 2 452 benchmark queries
- ровно одна строка для каждого `query_id`
- корректная длина `query_id` - 16 символов
- не более 50 `item_id` на запрос
- корректный формат `item_id` - 16 символов `0-9a-f`
- отсутствие неизвестных `item_id`
- отсутствие повторяющихся `item_id` внутри ответа одного запроса
- отсутствие запросов без кандидатов

Финальный submission содержит:

```text
benchmark queries:             2 452
submission rows:               2 452
unique query_id:               2 452
missing queries:                   0
candidates per query:             50
unknown item_id:                   0
duplicate item_id:                 0
invalid item_id format:            0
invalid query_id length:           0
```

---

## Возможные улучшения

При наличии дополнительного времени candidate generation можно расширить:

- semantic retrieval с использованием sentence embeddings
- более точным использованием исторических query-item взаимодействий
- комбинацией Word и Character TF-IDF
- отдельным retrieval по локации с последующим объединением candidate sets
- объединением кандидатов нескольких независимых retrieval-моделей
- подбором размера candidate pool и location boost на более репрезентативной validation-схеме

---

## Автор

Екатерина Акименко
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Конфигурация
TOP_K = 50
CANDIDATE_K = 500
BATCH_SIZE = 128
LOCATION_BOOST = 0.20



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
OUTPUT_PATH = OUTPUT_DIR / "answer.csv"


def build_item_text(items: pd.DataFrame) -> pd.Series:
    # Формирует текстовое представление объявления.

    # По результатам локальной validation лучшим вариантом оказался:
    # title + params + description.

    return (
        items["item_title_raw"].fillna("").astype(str)
        + " "
        + items["item_infm_params_text"].fillna("").astype(str)
        + " "
        + items["item_description_raw"].fillna("").astype(str)
    ).str.lower()


def build_query_text(queries: pd.DataFrame) -> pd.Series:
    # Формирует текстовое представление поискового запроса.

    return (
        queries["search_query"]
        .fillna("")
        .astype(str)
        .str.lower()
    )


def retrieve_candidates(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    batch_size: int = BATCH_SIZE,
    location_boost: float = LOCATION_BOOST,
) -> dict:
    # Генерирует Top-K кандидатов для каждого benchmark-запроса.

    # Алгоритм:
    # 1. Строит Word TF-IDF по title + params + description объявления.
    # 2. Считает cosine similarity между запросом и объявлениями.
    #    При L2-нормализованном TF-IDF это обычное скалярное произведение.
    # 3. Для каждого запроса оставляет расширенный набор lexical candidates.
    # 4. Добавляет bonus объявлениям из той же локации.
    # 5. Возвращает Top-K item_id.

    item_texts = build_item_text(items)
    query_texts = build_query_text(queries)

    print("Построение TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Обучаем TF-IDF только на benchmark corpus.
    item_matrix = vectorizer.fit_transform(item_texts)
    query_matrix = vectorizer.transform(query_texts)

    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_):,}")
    print(f"Item matrix: {item_matrix.shape}")
    print(f"Query matrix: {query_matrix.shape}")

    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()

    query_ids = queries["query_id"].to_numpy()
    query_locations = queries["search_location_id"].to_numpy()

    predictions = {}

    print("Поиск кандидатов...")

    for start in tqdm(
        range(0, query_matrix.shape[0], batch_size),
        desc="Retrieval",
    ):
        end = min(
            start + batch_size,
            query_matrix.shape[0],
        )

        # Sparse matrix:
        # batch_queries x all benchmark items
        similarities = (
            query_matrix[start:end]
            @ item_matrix.T
        )

        for local_idx in range(similarities.shape[0]):
            global_idx = start + local_idx

            query_id = query_ids[global_idx]
            query_location = query_locations[global_idx]

            row = similarities.getrow(local_idx)

            # Если TF-IDF не нашёл совпадений,
            # временно оставляем пустой список.
            # Ниже такие запросы получат fallback.
            if row.nnz == 0:
                predictions[query_id] = []
                continue

            scores = row.data
            indices = row.indices

            # Берём более широкий lexical candidate set,
            # чтобы location boost мог изменить итоговый Top-50.
            if len(scores) > candidate_k:
                candidate_positions = np.argpartition(
                    scores,
                    -candidate_k,
                )[-candidate_k:]
            else:
                candidate_positions = np.arange(
                    len(scores)
                )

            candidate_indices = indices[
                candidate_positions
            ]

            candidate_scores = scores[
                candidate_positions
            ].copy()

            candidate_locations = item_locations[
                candidate_indices
            ]

            # Мягкий bonus за совпадение локации.
            location_match = (
                candidate_locations == query_location
            ).astype(np.float32)

            final_scores = (
                candidate_scores
                + location_boost * location_match
            )

            # Итоговый Top-K.
            if len(final_scores) > top_k:
                top_positions = np.argpartition(
                    final_scores,
                    -top_k,
                )[-top_k:]
            else:
                top_positions = np.arange(
                    len(final_scores)
                )

            # Сортировка не влияет на Recall@50,
            # но делает результат детерминированным и удобным.
            top_positions = top_positions[
                np.argsort(
                    final_scores[top_positions]
                )[::-1]
            ]

            final_indices = candidate_indices[
                top_positions
            ]

            predictions[query_id] = (
                item_ids[final_indices].tolist()
            )

    return predictions


def fill_missing_candidates(
    predictions: dict,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict:
    # Гарантирует до Top-K кандидатов для каждого запроса.

    # Если lexical retrieval вернул меньше 50 объявлений,
    # сначала добавляются объявления из той же локации, затем любые объявления из benchmark corpus.

    # Fallback особенно важен для запросов без известных TF-IDF токенов.

    all_item_ids = items["item_id"].to_numpy()

    # Предварительно группируем объявления по локации.
    items_by_location = {
        location: group["item_id"].to_numpy()
        for location, group in items.groupby(
            "item_location_id",
            sort=False,
        )
    }

    for row in queries.itertuples(index=False):
        query_id = row.query_id
        query_location = row.search_location_id

        current = predictions.get(query_id, [])

        # Убираем возможные дубликаты,
        # сохраняя исходный порядок.
        current = list(dict.fromkeys(current))
        used = set(current)

        # Сначала fallback из той же локации.
        location_items = items_by_location.get(
            query_location,
            [],
        )

        for item_id in location_items:
            if len(current) >= top_k:
                break

            if item_id not in used:
                current.append(item_id)
                used.add(item_id)

        # Если всё ещё меньше Top-K —
        # заполняем любыми объявлениями corpus.
        if len(current) < top_k:
            for item_id in all_item_ids:
                if len(current) >= top_k:
                    break

                if item_id not in used:
                    current.append(item_id)
                    used.add(item_id)

        predictions[query_id] = current[:top_k]

    return predictions


def create_submission(
    predictions: dict,
    queries: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    # Создаёт answer.csv.

    # Одна строка соответствует одному query_id.
    # До 50 item_id записываются в одной строке
    # через один пробел.

    rows = []

    for query_id in queries["query_id"]:
        # приводим item_id к строкам.
        item_ids = [
            str(item_id)
            for item_id in predictions.get(query_id, [])[:TOP_K]
        ]

        rows.append(
            {
                "query_id": str(query_id),
                "item_id": " ".join(item_ids),
            }
        )

    submission = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    return submission


def validate_submission(
    submission: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
) -> None:
    # Проверяет формат финального answer.csv.

    expected_queries = set(
        queries["query_id"].astype(str)
    )

    actual_queries = set(
        submission["query_id"].astype(str)
    )

    valid_items = set(
        items["item_id"].astype(str)
    )

    invalid_items = 0
    duplicate_items = 0
    too_many = 0
    empty_queries = 0

    candidate_counts = []

    for row in submission.itertuples(index=False):
        item_ids = str(row.item_id).split()

        candidate_counts.append(len(item_ids))

        if len(item_ids) == 0:
            empty_queries += 1

        if len(item_ids) > TOP_K:
            too_many += 1

        # Повторы внутри одного ответа.
        duplicate_items += (
            len(item_ids) - len(set(item_ids))
        )

        # Проверяем, что каждый item_id существует
        # в benchmark corpus.
        invalid_items += sum(
            item_id not in valid_items
            for item_id in item_ids
        )

    print("\nПроверка submission")
    print("-" * 40)

    print(
        "Количество benchmark queries:",
        len(expected_queries),
    )

    print(
        "Количество строк submission:",
        len(submission),
    )

    print(
        "Уникальных query_id:",
        len(actual_queries),
    )

    print(
        "Пропущено queries:",
        len(expected_queries - actual_queries),
    )

    print(
        "Минимум candidates/query:",
        min(candidate_counts),
    )

    print(
        "Максимум candidates/query:",
        max(candidate_counts),
    )

    print(
        "Неизвестных item_id:",
        invalid_items,
    )

    print(
        "Повторов item_id внутри query:",
        duplicate_items,
    )

    assert len(submission) == len(queries), (
        "Количество строк должно совпадать "
        "с количеством benchmark queries."
    )

    assert expected_queries == actual_queries, (
        "Не все benchmark query_id присутствуют."
    )

    assert len(actual_queries) == len(submission), (
        "В submission есть повторяющиеся query_id."
    )

    assert too_many == 0, (
        "Для некоторых запросов больше 50 кандидатов."
    )

    assert empty_queries == 0, (
        "Для некоторых запросов нет кандидатов."
    )

    assert duplicate_items == 0, (
        "Внутри некоторых ответов повторяются item_id."
    )

    assert invalid_items == 0, (
        "Есть item_id, отсутствующие в benchmark corpus."
    )

    print("\nВсе проверки пройдены ✓")


def main():
    print("Загрузка benchmark данных...")

    queries = pd.read_parquet(
        QUERIES_PATH
    )

    items = pd.read_parquet(
        ITEMS_PATH
    )

    print(f"Queries: {queries.shape}")
    print(f"Items:   {items.shape}")

    predictions = retrieve_candidates(
        queries=queries,
        items=items,
    )

    predictions = fill_missing_candidates(
        predictions=predictions,
        queries=queries,
        items=items,
    )

    submission = create_submission(
        predictions=predictions,
        queries=queries,
        output_path=OUTPUT_PATH,
    )

    validate_submission(
        submission=submission,
        queries=queries,
        items=items,
    )

    print(
        f"\nSubmission сохранён в:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
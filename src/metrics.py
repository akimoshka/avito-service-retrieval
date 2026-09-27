def recall_at_k(predictions, relevant_items, k=50):
    # Вычисляет Recall@K для одного поискового запроса
    relevant = set(relevant_items)

    if not relevant:
        return 0.0

    predicted = set(predictions[:k])

    return len(predicted & relevant) / len(relevant)

def mean_recall_at_k(predictions, labels, k=50):
    # Вычисляет средний Recall@K по всем поисковым контекстам
    recalls = []

    for context_id, relevant_items in labels.items():
        predicted_items = predictions.get(context_id, [])

        recalls.append(
            recall_at_k(
                predicted_items,
                relevant_items,
                k=k,
            )
        )

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)
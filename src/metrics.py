def recall_at_k(predictions, relevant_items, k=50):
    """
    Узнаем Recall@K для одного запроса
    """
    relevant = set(relevant_items)

    if not relevant:
        return 0.0

    predicted = set(predictions[:k])

    return len(predicted & relevant) / len(relevant)
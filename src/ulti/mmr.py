import numpy as np

def apply_mmr(query_embedding, doc_embeddings, top_k=5, lambda_param=0.5):
    """Chọn top_k tài liệu dựa trên Maximal Marginal Relevance (MMR)"""
    selected = []
    candidate_indices = list(range(len(doc_embeddings)))

    similarity_to_query = np.dot(doc_embeddings, query_embedding)

    for _ in range(top_k):
        if not candidate_indices:
            break

        if not selected:
            idx = np.argmax(similarity_to_query)
        else:
            max_sim_to_selected = np.max(
                np.dot(doc_embeddings[selected], doc_embeddings.T), axis=0
            )
            max_sim_to_selected = max_sim_to_selected[candidate_indices]  # Ensure valid indices

            mmr_score = lambda_param * similarity_to_query[candidate_indices] - (1 - lambda_param) * max_sim_to_selected
            idx = candidate_indices[np.argmax(mmr_score)]  # Pick valid candidate index

        selected.append(idx)
        candidate_indices.remove(idx)
    return selected
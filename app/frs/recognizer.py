import numpy as np
from app.frs.gallery import load_all_embeddings


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def recognize_face(query_embedding, threshold=0.5):
    gallery = load_all_embeddings()

    best_name = "Unknown"
    best_score = 0.0

    for name, embeddings in gallery.items():
        for emb in embeddings:
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score < threshold:
        return "Unknown", float(best_score)

    return best_name, float(best_score)

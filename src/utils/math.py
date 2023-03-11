import numpy as np


def cosine_similarity(a: list[float] | None, b: list[float] | None):
    if a is None or b is None or not len(a) or not len(b):
        return 0

    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return min(max(0, similarity), 1)

from sklearn.metrics.pairwise import cosine_similarity

def detect_drift(embeddings, threshold=0.6):

    drift_points = []
    similarities = []

    for i in range(1, len(embeddings)):

        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[i-1]]
        )[0][0]

        similarities.append(sim)

        if sim < threshold:
            drift_points.append(i)

    return drift_points, similarities
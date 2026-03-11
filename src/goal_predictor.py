from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

goal_templates = {
    "Seek Medical Help": "I need medical treatment",
    "Buy Product": "I want to purchase something",
    "Book Service": "I want to book a service",
    "Learn Topic": "I want to learn something",
    "Solve Problem": "I need help solving a problem"
}

goal_embeddings = model.encode(list(goal_templates.values()))


def predict_goal(messages):

    embeddings = model.encode(messages)

    last_messages = embeddings[-3:]

    avg_embedding = np.mean(last_messages, axis=0)

    sims = cosine_similarity([avg_embedding], goal_embeddings)[0]

    index = sims.argmax()

    goal = list(goal_templates.keys())[index]

    return goal
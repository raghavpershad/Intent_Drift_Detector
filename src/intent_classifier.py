from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

intent_templates = {
    "Information Seeking": "tell me about something",
    "Concern": "is this dangerous",
    "Comparison": "which is better",
    "Decision Making": "should I choose this",
    "Action Seeking": "what should I do"
}

intent_embeddings = model.encode(list(intent_templates.values()))


def classify_intent(messages):

    message_embeddings = model.encode(messages)

    intents = []

    for emb in message_embeddings:

        sims = cosine_similarity([emb], intent_embeddings)[0]

        index = sims.argmax()

        intent = list(intent_templates.keys())[index]

        intents.append(intent)

    return intents
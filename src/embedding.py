from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(messages):
    """
    Convert list of sentences into embeddings
    """
    embeddings = model.encode(messages)
    return embeddings
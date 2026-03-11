# 🧠 Intent Drift Detector

Analyze how user intent evolves during a conversation using NLP embeddings.

## About the Project
Intent Drift Detector is a tool that analyzes conversations line-by-line to detect changes in user intent (topic drift), map user messages to specific intents, predict the user's final goal, and track the conversational flow through semantic similarity.

This project is especially useful for analyzing chatbots, customer support interactions, or any sequential text data where understanding the underlying goals of a user is important.

## How it Uses Tech 🛠️
This project leverages modern NLP and machine learning techniques:

* **[Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`)**: Used as the core embedding model. It converts each text message into a dense vector (embedding) that captures semantic meaning.
* **[Scikit-Learn](https://scikit-learn.org/) (`cosine_similarity`)**: 
  * **Drift Detection**: Calculates the similarity between the embedding of the current message and the previous one. A drop in similarity below a threshold indicates a drift in intent.
  * **Intent Classification**: Compares user message embeddings against predefined "intent templates" and classifies them based on the highest cosine similarity.
  * **Goal Prediction**: Compares the average embedding of the last few messages against "goal templates" to anticipate the final objective.
* **[NumPy](https://numpy.org/)**: Calculates the mean embeddings for goal prediction.
* **[Streamlit](https://streamlit.io/)**: Provides a fast, interactive web UI to input conversations and view the analysis.
* **[Matplotlib](https://matplotlib.org/)**: Visualizes the flow of semantic similarity across the conversation, making it easy to spot sudden drops (intent drifts).

## Features
- **Intent Classification** (e.g., Information Seeking, Concern, Action Seeking)
- **Intent Drift Detection** (Identifies exact points in the conversation where the topic shifted)
- **Goal Prediction** (Predicts the final end-goal of the user)
- **Similarity Visualization** (Graphs the semantic similarity of the conversation flow)

## Installation
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```
Then, enter your sample conversation (one sentence per line) into the web interface to view the analysis!

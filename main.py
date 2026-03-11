import json

from src.embedding import get_embeddings
from src.drift_detection import detect_drift
from src.intent_classifier import classify_intent
from src.goal_predictor import predict_goal


def analyze_conversation(messages):

    print("\nConversation:\n")

    for i, msg in enumerate(messages):
        print(f"{i+1}. {msg}")

    # Generate embeddings
    embeddings = get_embeddings(messages)

    # Detect intent drift
    drift_points, similarities = detect_drift(embeddings)

    # Classify intents
    intents = classify_intent(messages)

    # Predict final goal
    goal = predict_goal(messages)

    print("\nDetected Intents:\n")

    for i in range(len(messages)):
        print(f"{messages[i]}  →  {intents[i]}")

    print("\nSimilarity Scores Between Messages:")
    print(similarities)

    if drift_points:
        print("\nIntent Drift Detected At Message Index:", drift_points)
    else:
        print("\nNo Major Intent Drift Detected")

    print("\nPredicted Final Goal:", goal)


def main():

    with open("data/conversations.json") as f:
        data = json.load(f)

    # Select first conversation for demo
    conversation = data[0]["conversation"]

    analyze_conversation(conversation)


if __name__ == "__main__":
    main()
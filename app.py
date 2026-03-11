import streamlit as st
import matplotlib.pyplot as plt

from src.embedding import get_embeddings
from src.drift_detection import detect_drift
from src.intent_classifier import classify_intent
from src.goal_predictor import predict_goal


st.set_page_config(page_title="Intent Drift Detector", layout="centered")

st.title("🧠 Intent Drift Detector")
st.write("Analyze how user intent evolves during a conversation using NLP embeddings.")

st.write("Enter conversation (one sentence per line):")

text = st.text_area(
    "Conversation",
    height=200,
    placeholder="Example:\nI have a headache\nIs it serious\nCould it be migraine\nWhich doctor should I see"
)

if st.button("Analyze Conversation"):

    if text.strip() == "":
        st.warning("Please enter a conversation.")
    else:

        messages = [line.strip() for line in text.split("\n") if line.strip()]

        # Generate embeddings
        embeddings = get_embeddings(messages)

        # Detect drift
        drift_points, similarities = detect_drift(embeddings)

        # Classify intents
        intents = classify_intent(messages)

        # Predict goal
        goal = predict_goal(messages)

        st.subheader("Detected Intents")

        for i in range(len(messages)):
            st.write(f"**{messages[i]}** → {intents[i]}")

        st.subheader("Intent Drift Points")

        if drift_points:
            st.write(f"Drift detected at message index: {drift_points}")
        else:
            st.write("No significant drift detected.")

        st.subheader("Predicted Final Goal")
        st.success(goal)

        st.subheader("Similarity Between Messages")

        if similarities:

            fig = plt.figure()

            plt.plot(range(1, len(similarities)+1), similarities, marker="o")

            plt.title("Conversation Similarity Flow")
            plt.xlabel("Message Transition")
            plt.ylabel("Semantic Similarity")

            st.pyplot(fig)
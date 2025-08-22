import random
import json
import pickle
import nltk
from nltk.stem import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()

# Load the intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load the trained model and vectorizer
with open('chatbot_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def preprocess_sentence(sentence):
    """Tokenizes and stems the words in the sentence."""
    words = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w.lower()) for w in words]
    return " ".join(stemmed_words)

def get_response(sentence):
    """Predicts the intent and returns a random response."""
    preprocessed_sentence = preprocess_sentence(sentence)
    X = vectorizer.transform([preprocessed_sentence])
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X)
    max_prob = max(probabilities[0])

    # If confidence is high, return the predicted tag's response
    if max_prob > 0.5:  # Confidence threshold
        prediction = model.predict(X)[0]
        for intent in intents['intents']:
            if intent['tag'] == prediction:
                return random.choice(intent['responses'])
    else:
        return "I'm not sure I understand. Could you rephrase?"

# Main chat loop
if __name__ == "__main__":
    print("Start chatting with the bot! (type 'quit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        response = get_response(user_input)
        print(f"Bot: {response}")
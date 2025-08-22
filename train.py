import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download NLTK data (only need to do this once)
nltk.download('punkt')
nltk.download('punkt_tab') # <-- ADD THIS LINE

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# --- 1. Preprocessing the data ---
words = []
tags = []
patterns = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        patterns.append((wrds, intent['tag']))
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

# Stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

# --- 2. Create Training Data ---
training = []
output = []

for (pattern_words, tag) in patterns:
    # Stem each word
    stemmed_words = [stemmer.stem(word.lower()) for word in pattern_words]
    training.append(" ".join(stemmed_words))
    output.append(tag)

print("Training data created.")

# --- 3. Vectorize the text ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training)
y = np.array(output)

# --- 4. Train the Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model training complete.")

# --- 5. Save the Model and Vectorizer ---
with open('chatbot_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved.")
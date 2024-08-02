from flask import Flask, render_template, request, jsonify
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load intents and model data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_intent_by_cosine_similarity(sentence, intents):
    sentence_tokens = tokenize(sentence)
    sentence_bag = bag_of_words(sentence_tokens, all_words)
    sentence_bag = sentence_bag.reshape(1, -1)
    
    best_match = None
    highest_similarity = 0
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = tokenize(pattern)
            pattern_bag = bag_of_words(pattern_tokens, all_words)
            pattern_bag = pattern_bag.reshape(1, -1)
            
            similarity = cosine_similarity(sentence_bag, pattern_bag)[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = intent

    return best_match

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    intent = get_intent_by_cosine_similarity(user_message, intents)
    
    if intent:
        response = random.choice(intent['responses'])
    else:
        response = "I do not understand..."
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()

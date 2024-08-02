import random
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

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

print("\nLet's chat! (type 'quit' to exit)")

while True:
    sentence = input("\nYou: ")
    if sentence == "quit":
        break

    start_time = time.time()
    
    intent = get_intent_by_cosine_similarity(sentence, intents)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    if intent:
        print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
    
    print(f"Inference time: {inference_time:.4f} seconds")

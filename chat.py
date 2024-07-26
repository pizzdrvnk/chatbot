import random
import json
import requests

import torch

import mysql.connector

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dataset.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "dataset.pth"
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

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="skripsi"
    )

def load_intens(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents['intents']

def find_intent(intens, user_input):
    for intent in intens:
        for pattern in intent['patterns']:
            if pattern in user_input.lower():
                return intent
    return None

def get_response_and_save(user_input):
    intents = load_intens('dataset.json')
    matched_intent = find_intent(intents, user_input)

    if matched_intent:
        tag = matched_intent['tag']
        response = random.choice(matched_intent['responses'])
    else:
        tag = "unknown"
        response = "Mohon maaf, kami belum bisa membantu. Anda bisa ketik '0' untuk menu atau sampaikan pertanyaan lain seputar produk atau layanan kami.\nUntuk info lebih lanjut dapat menghubungi admin di WhatsApp : <a href='https://wa.me/6285802485532' target='_blank'>085802485532.</a>"
    
    # Simpan ke database
    db_connection = connect_db()
    cursor = db_connection.cursor()
    sql = "INSERT INTO respon_bot (tag, response) VALUES (%s, %s)"
    values = (tag, response)
    cursor.execute(sql, values)
    db_connection.commit()
    cursor.close()
    db_connection.close()

    return response

def send_response_to_laravel(response):
    url = 'http://127.0.0.1:8000/predict'
    data = {
        'response': response
    }
    headers = {
        'Content-Type': 'application/json'
        }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print('Response stored successfully')
    else:
        print('Failed to store response', response.text)


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return response, None
    
    return "Mohon maaf, kami belum bisa membantu. Anda bisa ketik '0' untuk menu atau sampaikan pertanyaan lain seputar produk atau layanan kami.\nUntuk info lebih lanjut dapat menghubungi admin di WhatsApp : <a href='https://wa.me/6285802485532' target='_blank'>085802485532.</a>"

def welcome_message():
    return (
        "Selamat datang di layanan Jaya Custom! Bagaimana kami bisa membantu Anda hari ini?\n"
        "Ketik angka 0 untuk menampilkan menu."
    )

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    # Tampilkan pesan sambutan
    welcome = welcome_message()
    print(welcome)
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp, links = get_response(sentence)
        print(resp)
        

# Inisialisasi model dan data untuk chatbot
# def initialize_chatbot():
#     FILE = "data.pth"
#     data = torch.load(FILE)

#     input_size = data["input_size"]
#     hidden_size = data["hidden_size"]
#     output_size = data["output_size"]
#     all_words = data['all_words']
#     tags = data['tags']
#     model_state = data["model_state"]

#     model = NeuralNet(input_size, hidden_size, output_size).to(device)
#     model.load_state_dict(model_state)
#     model.eval()

#     return model, all_words, tags

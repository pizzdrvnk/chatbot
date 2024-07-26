from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response_and_save

app = Flask(__name__)
CORS(app) 


def welcome_message():
    return (
        "Selamat datang di layanan Jaya Custom! Bagaimana kami bisa membantu Anda hari ini?\n"
        "Ketik angka 0 untuk menampilkan menu."
    )

# @app.get('/')
# def index():
#     return render_template('base.html')

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    #  TODO: check if text is valid
    response = get_response_and_save(text)
    message = {"answer": response}
    return jsonify(message)

# @app.post('/predict')
# def predict():
#     text = request.get_json().get("message")
#     #  TODO: check if text is valid
#     response = get_response(text)
#     message = {"answer": response}
#     return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
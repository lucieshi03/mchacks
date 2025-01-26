from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, methods=["GET", "POST", "OPTIONS"])

app.route('/post', methods=['POST'])
# Sends information to OpenAi

def handle_post():
    if request.method == 'POST':
        return

@app.route('/get', methods=['GET'])
def handle_get():
    if request.method == 'GET':
        return
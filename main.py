from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask "})
@app.route('/post_example', methods=['POST'])
def post_example():
    if request.method == 'POST':
        # Access the data sent in the POST request
        data = request.get_json()  # Assuming the data is sent as JSON
        # You can also use request.form to get form data
        # data = request.form

        # Process the data
        # For example, if the JSON contains a key 'message'
        if 'question' in data:
            received_message = data['question']
            #o=predict(received_message)

            #return f"Received message: {o}"
            return jsonify({"Received message": ok})
        else:
            return "No 'message' key found in the POST request data"
    else:
        return "This endpoint only accepts POST requests"


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

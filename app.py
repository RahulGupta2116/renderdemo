from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from models.model_def import MyModel  # Import the model class
import pickle

# Load tokenizer (Change model name if needed)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Initialize the model (Ensure architecture matches saved weights)
model = MyModel()
model.load_state_dict(torch.load('demo_model.pkl', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

print("Loading models...")
with open("hatespeech/saved_models/lr_model.pkl", 'rb') as f:
    hate_model = pickle.load(f)
with open("hatespeech/saved_models/vectorizer.pkl", 'rb') as f:
    hate_vect = pickle.load(f)

def ishate(string: str) -> bool:
    string = [string]
    sen_trans = hate_vect.transform(string)
    prediction = hate_model.predict(sen_trans)[0]  # 0->normal 1->toxic
    return bool(prediction)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/check_hate', methods=['POST'])
def check_hate():
    try:
        data = request.json
        received_text = data.get('text')
        if received_text:
            result = ishate(received_text)
            return jsonify({"result": result})
        else:
            return jsonify({"error": "No text received"}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
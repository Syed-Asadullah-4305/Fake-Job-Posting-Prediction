import os

# ===============================
# ENVIRONMENT SETTINGS (IMPORTANT)
# ===============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ===============================
# CONSTANTS
# ===============================
MAX_SEQUENCE_LENGTH = 200
THRESHOLD = 0.7

# ===============================
# LOAD TOKENIZER
# ===============================
try:
    print("Loading tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print("TOKENIZER LOAD ERROR:", str(e))
    tokenizer = None

# ===============================
# LOAD MODEL
# ===============================
try:
    print("Loading Keras model...")
    model = tf.keras.models.load_model("fake_job_lstm_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None

# ===============================
# PREPROCESSING FUNCTION
# ===============================
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )
    return padded

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check tokenizer/model availability
        if tokenizer is None or model is None:
            return render_template(
                "index.html",
                prediction="⚠️ Model resources failed to load on server."
            )

        combined_text = request.form.get("combined_text")

        if not combined_text or combined_text.strip() == "":
            return render_template(
                "index.html",
                prediction="❗ Please enter the job description."
            )

        print("Received input for prediction")

        # Preprocess input
        input_data = preprocess_text(combined_text)

        print("Running model prediction...")

        # Predict
        prediction_score = float(model.predict(input_data, verbose=0)[0][0])

        print("Prediction score:", prediction_score)

        # Classification
        result = (
            "Fraudulent 🚨"
            if prediction_score > THRESHOLD
            else "Legitimate ✅"
        )

        return render_template(
            "index.html",
            prediction=f"The job post is: {result}"
        )

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return render_template(
            "index.html",
            prediction="⚠️ Prediction failed. Please try again."
        )

# ===============================
# MAIN (LOCAL RUN ONLY)
# ===============================
if __name__ == "__main__":
    app.run(debug=True)

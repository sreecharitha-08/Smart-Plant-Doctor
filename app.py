from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import json
import cv2
import numpy as np
import random
from predict import predict_image
from werkzeug.utils import secure_filename
from chatbot import get_chatbot_response
from email_utils import send_evaluation_email

# ✅ Gemini Quiz Import
from gemini_quiz import generate_dynamic_questions

app = Flask(__name__)
app.secret_key = "smartplantdoctor"

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload Folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---------------- HELPER FUNCTIONS ----------------

def calculate_affected_area(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Healthy green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    total_pixels = 224 * 224
    affected_pixels = total_pixels - green_pixels
    affected_percentage = (affected_pixels / total_pixels) * 100

    return round(affected_percentage, 2)

def generate_disease_info(prediction, confidence):
    parts = prediction.split("_")
    plant = parts[0]
    severity = None

    if "healthy" in prediction.lower():
        status = "Healthy"
        summary = f"Your {plant} plant looks healthy."
        care = "Continue regular watering and provide proper sunlight."
        prevention = "Monitor plant weekly and maintain soil nutrition."
        severity = "None"
    elif "Early_blight" in prediction:
        status = "Diseased"
        summary = f"{plant} plant is affected by Early blight."
        care = "Remove infected leaves and apply fungicide."
        prevention = "Ensure good air circulation and avoid overwatering."
    elif "Late_blight" in prediction:
        status = "Diseased"
        summary = f"{plant} plant is affected by Late blight."
        care = "Apply copper-based fungicide immediately."
        prevention = "Avoid excess moisture and water at soil level."
    else:
        status = "Unknown"
        summary = "No detailed information available."
        care = "-"
        prevention = "-"
        severity = "Unknown"

    if status == "Diseased":
        if confidence < 70:
            severity = "Low"
        elif confidence < 85:
            severity = "Medium"
        else:
            severity = "High"

    return plant, status, summary, care, prevention, severity

def load_quiz_data():
    quiz_path = os.path.join(BASE_DIR, "quiz_data.json")
    try:
        with open(quiz_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"level_1": []}

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask_bot():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        reply = get_chatbot_response(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/quiz")
def quiz():
    # 1. Clear previous quiz session to ensure fresh start
    session.pop("current_questions", None)
    
    level = 1
    questions = []

    try:
        # 2. Attempt to get unique questions from Gemini
        questions = generate_dynamic_questions(level)
        if not questions or not isinstance(questions, list):
            raise Exception("Invalid data received from Gemini")
    except Exception as e:
        print(f"⚠ Gemini variety failed: {e}. Loading static questions.")
        # 3. Fallback: Pick 5 random questions from JSON to avoid repetition
        quiz_data = load_quiz_data()
        pool = quiz_data.get("level_1", [])
        if len(pool) >= 5:
            questions = random.sample(pool, 5)
        else:
            questions = pool

    session["current_questions"] = questions
    return render_template("quiz.html", questions=questions, level=level)

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    questions = session.get("current_questions", [])
    score = 0
    total = len(questions)
    results = []

    for i, q in enumerate(questions):
        selected = request.form.get(f"q{i}")
        correct = q["answer"]
        is_correct = (selected == correct)
        
        if is_correct:
            score += 1

        results.append({
            "question": q["question"],
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct
        })

    return render_template("result.html", 
                           score=score, 
                           total=total, 
                           level=1, 
                           unlocked=False, 
                           results=results)

@app.route("/detect", methods=["GET", "POST"])
def detect():
    prediction = None
    confidence = None
    image_path = None
    plant = status = summary = care = prevention = severity = disease_name = affected_area = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)

            prediction, confidence = predict_image(full_path)
            plant, status, summary, care, prevention, severity = generate_disease_info(prediction, confidence)

            if status == "Diseased":
                disease_name = prediction.replace("_", " ")
                affected_area = calculate_affected_area(full_path)

            # Change to your desired notification email
            receiver_email = "Your_mail"
            try:
                send_evaluation_email(
                    receiver_email=receiver_email,
                    plant=plant,
                    status=status,
                    severity=severity,
                    confidence=round(confidence, 2) if confidence else 0,
                    summary=summary,
                    care=care,
                    prevention=prevention,
                    disease_name=disease_name,
                    affected_area=affected_area
                )
            except Exception as e:
                print(f"Email sending failed: {e}")

            image_path = os.path.join("static", "uploads", filename)

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        image_path=image_path,
        plant=plant,
        status=status,
        summary=summary,
        care=care,
        prevention=prevention,
        severity=severity,
        disease_name=disease_name,
        affected_area=affected_area
    )

if __name__ == "__main__":
    app.run(debug=True)
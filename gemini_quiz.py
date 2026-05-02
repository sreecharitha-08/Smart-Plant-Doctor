#gemini_quiz.py
import google.generativeai as genai
import json
import os

# Configure your API Key
genai.configure(api_key="google_gemini_api")

def generate_dynamic_questions(level):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Generate 5 multiple-choice questions about agriculture and plant health for "Level {level}".
    The difficulty should match the level (Level 1: Basic, Level 2: Intermediate, Level 3: Expert).
    python -m venv venv
    Return the response strictly in this JSON format:
    [
      {{
        "question": "Question text here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "The correct option text"
      }}
    ]
    Do not include markdown formatting like ```json.
    """
    
    response = model.generate_content(prompt)
    
    # Clean and parse the response
    try:
        # Removing potential markdown blocks if Gemini includes them
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        questions = json.loads(cleaned_response)
        return questions
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return [] # Fallback to empty list or static data
import google.generativeai as genai

# 🔑 PUT YOUR REAL API KEY HERE
API_KEY = "google_gemini_api"

# Configure Gemini API
genai.configure(api_key=API_KEY)

# ✅ Use officially supported model (works with v1beta)
MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)


def get_chatbot_response(user_message):
    try:
        if not user_message or user_message.strip() == "":
            return "Please ask a plant-related question 🌱"

        prompt = f"""
        You are a Smart Plant Doctor AI assistant.
        Only answer questions related to:
        - Plant diseases
        - Plant care
        - Fertilizers
        - Watering
        - Leaf problems
        
        Give clear, simple and accurate answers.
        
        User Question: {user_message}
        """

        response = model.generate_content(prompt)

        # Proper safe extraction (important)
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "Sorry, I couldn't generate a plant advice. Try asking in a full sentence 🌿"

    except Exception as e:
        return f"Chatbot Error: {str(e)}"
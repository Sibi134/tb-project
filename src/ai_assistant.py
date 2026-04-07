from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


def tb_chatbot(user_message):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI healthcare assistant explaining tuberculosis analytics and public health insights."
                },
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"


def explain_prediction(prediction_text):
    try:

        prompt = f"""
You are a public health analytics expert.

Analyze the following tuberculosis prediction output and generate a short professional
healthcare insight report suitable for a data dashboard.

Prediction Data:
{prediction_text}

Formatting rules:
- Use Markdown headings for section titles
- Do NOT use ** or star symbols
- Keep explanations concise (1–2 sentences)

Output structure:

### Predicted TB Burden
Explain what the predicted number of TB cases indicates.

### Hospital Capacity Requirement
Explain why hospital beds are required.

### Medical Workforce Requirement
Explain why doctors are needed.

### Diagnostic Infrastructure
Explain why TB test kits are required.

### Public Health Recommendation
Explain how health authorities should respond.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an epidemiology analytics expert."
                },
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"
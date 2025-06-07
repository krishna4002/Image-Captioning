import openai
import os
from dotenv import load_dotenv
load_dotenv()

def generate_caption_openrouter(prompt: str, api_key: str) -> str:
    # Configure OpenAI client
    API_KEY = os.getenv("OPEN_ROUTER_KEY")
    client = openai.OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",  # or any model you use via OpenRouter
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates image captions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


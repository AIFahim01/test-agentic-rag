from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Test OpenAI API call
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.7,
        max_tokens=100
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")

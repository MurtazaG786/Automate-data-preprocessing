import google.genai as genai
import os
from dotenv import load_dotenv
load_dotenv()

client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
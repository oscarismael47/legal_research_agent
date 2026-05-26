import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = ChatOpenAI(model="gpt-4.1", temperature=0.0, api_key=OPENAI_API_KEY)
from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
client = genai.Client()

def generate_response(query, top_db_responses, chat_history):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="You are a helpful assistant. According to the context: {top_db_responses}, answer the following question: {query}, you can use the chat history: {chat_history}".format(top_db_responses=top_db_responses, query=query, chat_history=chat_history),
    )
    return response.text
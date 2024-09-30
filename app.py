from dotenv import load_dotenv
import os
import google.generativeai as genai
from gtts import gTTS
import tempfile
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure the Google API with the provided key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

app = FastAPI()

# Pydantic model for input validation
class QuestionRequest(BaseModel):
    question: str

# Function to get the Gemini response and aggregate streamed chunks
def get_gemini_response(question):
    optimized_prompt = (
        f"Respond to this question by referencing the FSSAI (Food Safety and Standards Authority of India) website (https://fssai.gov.in) "
        f"or any other FSSAI-approved data sources. Treat all queries related to food safety, hygiene, regulations, food standards, "
        f"labeling, food inspections, food licensing, food businesses, and consumer safety in India as being within the FSSAI's scope, "
        f"even if the acronym 'FSSAI' is not mentioned. Provide detailed information or direct the user to relevant sections of the FSSAI website. "
        f"If no direct answer is available, give a general response about how FSSAI addresses such queries, and encourage the user to visit the website "
        f"for more information. Here is the question: {question}"
    )
    response_chunks = chat.send_message(optimized_prompt, stream=True)
    full_response = "".join([chunk.text for chunk in response_chunks])
    return full_response

# Text-to-speech function
def text_to_speech(text):
    clean_text = re.sub(r'[*_~`]', '', text)
    clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
    tts = gTTS(text=clean_text, lang='en')
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    tts.save(temp_file_path)
    return temp_file_path

# FastAPI endpoint to get response
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        response = get_gemini_response(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint to convert text to speech
@app.post("/text-to-speech")
async def tts_endpoint(request: QuestionRequest):
    try:
        audio_file_path = text_to_speech(request.question)
        return {"audio_file": audio_file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

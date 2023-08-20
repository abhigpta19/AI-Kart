import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseSettings

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

from app.models.schemas.email import Email
from app.utils import *
from app.utils import load_model, load_tokenizer, generate_text

from app.utils import *


class Settings(BaseSettings):
    model_path: str = 'suraj/'
    words = []
    characters = []


settings = Settings()
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/spam/predict")
async def get_prediction(email: Email):
    # email_features = get_email_features(email.text, settings.words, settings.characters).reshape(1, -1)
    generated_text = generate_text('suraj/', email.text, max_length=50)  # Generate text using the function  
    return {'label': generated_text } 



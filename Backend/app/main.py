from fastapi import FastAPI, status
from app.cors import setup_cors
from dotenv import load_dotenv
import os
from app.routers import ai_powered_document_search

load_dotenv()

app = FastAPI()
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
setup_cors(app, origins)


app.include_router(ai_powered_document_search.router, prefix="", tags=["AI Powered Document Search"])

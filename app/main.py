from dotenv import load_dotenv
load_dotenv()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware      
from app.api.endpoints import router as api_router
from app.routes import auth
from app.core.config import settings
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="A RAG‑powered API for interacting with documents."
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("APP_SECRET"),
    same_site="lax",
    https_only=False,
    max_age=600,
)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR) 

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Visit {settings.API_V1_STR}/docs for documentation."}

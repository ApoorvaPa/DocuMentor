from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="DocuMentor RAG Microservice",
    description="Answer document-grounded queries using RAG with PDF/JPG support",
    version="0.1.0"
)

# Include the routes from routes.py
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to DocuMentor 🚀"}

from fastapi import FastAPI
from app.api.recognize import router as recognize_router
from app.api.detect import router as detect_router
from app.api.add_identity import router as add_identity_router

app = FastAPI(title="Face Recognition System")

# register routes
app.include_router(detect_router)
app.include_router(add_identity_router)
app.include_router(recognize_router)


@app.get("/")
def root():
    return {"message": "FRS API is running 🚀"}

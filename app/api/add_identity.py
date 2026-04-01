from fastapi import APIRouter, UploadFile, File, Form
import cv2
import numpy as np

from app.frs.detector import detect_faces
from app.frs.gallery import save_embedding

router = APIRouter(prefix="/add-identity", tags=["Add Identity"])


@router.post("")
async def add_identity(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"error": "Invalid image"}

    faces = detect_faces(image)
    if len(faces) == 0:
        return {"error": "No face detected"}

    face = faces[0]
    embedding = face.embedding

    save_embedding(name, embedding, image)

    return {"message": "Identity saved successfully"}

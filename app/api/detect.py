from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.frs.detector import detect_faces

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = detect_faces(image)

    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.tolist(),
            "landmarks": face.kps.tolist(),
            "confidence": float(face.det_score)
        })

    return {"faces": results}

from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.frs.detector import detect_faces
from app.frs.gallery import load_all_embeddings

router = APIRouter(prefix="/recognize", tags=["Recognition"])


@router.post("")
async def recognize(file: UploadFile = File(...)):
    # -------- read image --------
    image_bytes = await file.read()
    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"results": []}

    # -------- detect faces --------
    faces = detect_faces(image)
    gallery = load_all_embeddings()

    results = []

    for face in faces:
        emb = face.embedding

        # normalize query embedding
        emb_norm = emb / np.linalg.norm(emb)

        best_name = "Unknown"
        best_score = 0.0

        for name, embeddings in gallery.items():
            for stored_emb in embeddings:
                # normalize stored embedding
                stored_norm = stored_emb / np.linalg.norm(stored_emb)

                # cosine similarity
                score = float(np.dot(emb_norm, stored_norm))

                # threshold tuned for InsightFace
                if score > best_score and score > 0.35:
                    best_score = score
                    best_name = name

        x1, y1, x2, y2 = map(int, face.bbox)
        landmarks = face.kps.astype(int).tolist()

        results.append({
            "name": best_name,
            "confidence": round(best_score * 100, 2),
            "bbox": [x1, y1, x2, y2],
            "landmarks": landmarks
        })

    return {"results": results}

from insightface.app import FaceAnalysis

# Initialize InsightFace once (global object)
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(image):
    """
    image: BGR image (OpenCV format)
    returns: list of InsightFace face objects
    """
    faces = face_app.get(image)
    return faces

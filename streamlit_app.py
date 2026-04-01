import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from app.frs.detector import detect_faces
from app.frs.gallery import load_all_embeddings, save_embedding

# ================= CONFIG =================
GALLERY_DIR = "app/data/gallery"

st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("Face Recognition System")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Face Detection",
    "🧠 Face Recognition",
    "➕ Add Identity",
    "📋 List Identities"
])


def load_image_bytes(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)


def draw_label(img, x1, y1, name, conf):
    label = f"{name} ({conf:.2f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    cv2.rectangle(
        img,
        (x1, y1 - th - 15),
        (x1 + tw + 10, y1),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        img,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def show_and_prepare_download(names):
    if not names:
        st.warning("No recognized faces were detected.")
        return

    st.subheader("Recognized names")
    for n in names:
        st.write("- ", n)

    text_content = "\n".join(names)
    save_path = "recognized_names.txt"

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        st.success(f"Recognized names also written to file: {save_path}")
    except Exception as e:
        st.error(f"Failed to write local file: {e}")

    st.download_button(
        label="Download recognized names",
        data=text_content,
        file_name="recognized_names.txt",
        mime="text/plain"
    )


def detect_image(image):
    faces = detect_faces(image)
    results = []
    for face in faces:
        results.append({
            "bbox": list(map(int, face.bbox)),
            "landmarks": face.kps.astype(int).tolist(),
            "confidence": float(face.det_score)
        })
    return results


def recognize_image(image):
    faces = detect_faces(image)
    gallery = load_all_embeddings()
    results = []

    for face in faces:
        emb = face.embedding
        emb_norm = emb / np.linalg.norm(emb)

        best_name = "Unknown"
        best_score = 0.0

        for name, embeddings in gallery.items():
            for stored_emb in embeddings:
                stored_norm = stored_emb / np.linalg.norm(stored_emb)
                score = float(np.dot(emb_norm, stored_norm))

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

    return results


def add_identity(name, image):
    if not name.strip():
        return {"error": "Please enter a name."}

    faces = detect_faces(image)
    if len(faces) == 0:
        return {"error": "No face detected."}

    face = faces[0]
    save_embedding(name.strip(), face.embedding, image)
    return {"message": "Identity saved successfully."}


# ======================================================
# 1️⃣ FACE DETECTION
# ======================================================
with tab1:
    st.subheader("Face Detection (Upload Image)")
    file = st.file_uploader("Upload image", ["jpg", "png", "jpeg"], key="detect")

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Input Image", width=350)

        if st.button("Detect Faces"):
            image = load_image_bytes(file.getvalue())
            if image is None:
                st.error("Could not read image file.")
            else:
                faces = detect_image(image)
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                for f in faces:
                    x1, y1, x2, y2 = map(int, f["bbox"])
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption="Detected Faces",
                    width=350
                )


# ======================================================
# 2️⃣ FACE RECOGNITION
# ======================================================
with tab2:
    mode = st.radio("Choose input method", ["Upload Image", "Camera Capture"])

    if mode == "Upload Image":
        file = st.file_uploader("Upload image", ["jpg", "png", "jpeg"], key="rec_upload")

        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Input Image", width=350)

            if st.button("Recognize"):
                image = load_image_bytes(file.getvalue())
                if image is None:
                    st.error("Could not read image file.")
                else:
                    data = recognize_image(image)
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    recognized_names = []

                    for r in data:
                        x1, y1, x2, y2 = map(int, r["bbox"])
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for (x, y) in r["landmarks"]:
                            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 255), -1)

                        name = r["name"].strip()
                        draw_label(img_bgr, x1, y1, name, r["confidence"])

                        if name and name.lower() not in ["unknown", "not recognized", "no match"]:
                            recognized_names.append(f"{name} ({r['confidence']:.2f}%)")

                    st.image(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        caption="Recognition Result",
                        width=350
                    )

                    show_and_prepare_download(recognized_names)

    else:
        cam = st.camera_input("Capture image from camera")

        if cam:
            img = Image.open(cam).convert("RGB")
            st.image(img, caption="Captured Image", width=350)

            if st.button("Recognize Face"):
                image = load_image_bytes(cam.getvalue())
                if image is None:
                    st.error("Could not read camera image.")
                else:
                    data = recognize_image(image)
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    recognized_names = []

                    for r in data:
                        x1, y1, x2, y2 = map(int, r["bbox"])
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for (x, y) in r["landmarks"]:
                            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 255), -1)

                        name = r["name"].strip()
                        draw_label(img_bgr, x1, y1, name, r["confidence"])

                        if name and name.lower() not in ["unknown", "not recognized", "no match"]:
                            recognized_names.append(f"{name} ({r['confidence']:.2f}%)")

                    st.image(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        caption="Recognition Result",
                        width=350
                    )

                    show_and_prepare_download(recognized_names)


# ======================================================
# 3️⃣ ADD IDENTITY
# ======================================================
with tab3:
    mode = st.radio("Add Identity Using", ["Upload Image", "Camera Capture"])
    name = st.text_input("Enter person name")

    if mode == "Upload Image":
        file = st.file_uploader("Upload face image", ["jpg", "png", "jpeg"], key="add_upload")

        if file:
            st.image(file, caption="Input Image", width=300)

            if st.button("Save Identity"):
                image = load_image_bytes(file.getvalue())
                if image is None:
                    st.error("Could not read image file.")
                else:
                    result = add_identity(name, image)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(result["message"])

    else:
        cam = st.camera_input("Capture face image", key="add_camera")

        if cam:
            st.image(cam, caption="Captured Image", width=300)

            if st.button("Save Identity"):
                image = load_image_bytes(cam.getvalue())
                if image is None:
                    st.error("Could not read camera image.")
                else:
                    result = add_identity(name, image)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(result["message"])


# ======================================================
# 4️⃣ LIST IDENTITIES
# ======================================================
with tab4:
    st.subheader("Registered Identities")

    if os.path.exists(GALLERY_DIR):
        persons = sorted(os.listdir(GALLERY_DIR))
        if persons:
            for person in persons:
                col1, col2 = st.columns([1, 4])
                img_path = os.path.join(GALLERY_DIR, person, "reference.jpg")

                with col1:
                    if os.path.exists(img_path):
                        st.image(img_path, width=80)

                with col2:
                    st.markdown(f"**{person}**")
        else:
            st.info("No identities found")
    else:
        st.info("No identities found")

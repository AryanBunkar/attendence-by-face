import os
import numpy as np
import cv2

# ONE SINGLE GALLERY LOCATION (LOCKED)
BASE_DIR = "app/data/gallery"


def save_embedding(name, embedding, image=None):
    """
    Saves embedding(s) and one reference image per identity.
    - embeddings.npy  (multiple embeddings)
    - reference.jpg   (saved once)
    """

    person_dir = os.path.join(BASE_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # ---------- save embeddings ----------
    emb_path = os.path.join(person_dir, "embeddings.npy")

    if os.path.exists(emb_path):
        old_embeddings = np.load(emb_path)
        embeddings = np.vstack([old_embeddings, embedding])
    else:
        embeddings = np.array([embedding])

    np.save(emb_path, embeddings)

    # ---------- save reference image (only once) ----------
    if image is not None:
        ref_path = os.path.join(person_dir, "reference.jpg")
        if not os.path.exists(ref_path):
            cv2.imwrite(ref_path, image)


def load_all_embeddings():
    """
    Loads all embeddings from gallery.
    Returns:
        dict { person_name : np.ndarray (N, 512) }
    """

    gallery = {}

    if not os.path.exists(BASE_DIR):
        return gallery

    for person in os.listdir(BASE_DIR):
        emb_path = os.path.join(BASE_DIR, person, "embeddings.npy")
        if os.path.exists(emb_path):
            gallery[person] = np.load(emb_path)

    return gallery

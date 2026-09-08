from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "gender_cnn.keras"
FACE_DETECTOR_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
CLASS_NAMES = ["female", "male"]
IMG_SIZE = 128
FACE_SCORE_THRESHOLD = 0.85


st.set_page_config(page_title="Face Label Classifier", page_icon="📷", layout="centered")


@st.cache_resource
def load_classifier():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    if not FACE_DETECTOR_PATH.is_file():
        raise FileNotFoundError(f"Face detector not found: {FACE_DETECTOR_PATH}")
    return cv2.FaceDetectorYN.create(
        str(FACE_DETECTOR_PATH), "", (320, 320), FACE_SCORE_THRESHOLD, 0.3, 5000
    )


def classify_faces(image_bgr, model, detector):
    height, width = image_bgr.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(image_bgr)
    faces = [] if detections is None else [detection for detection in detections if detection[14] >= FACE_SCORE_THRESHOLD]

    if len(faces) == 0:
        return image_bgr, []

    crops = []
    boxes = []
    for detection in faces:
        x, y, face_width, face_height = detection[:4].astype(int)
        x = max(0, x)
        y = max(0, y)
        face_width = min(face_width, width - x)
        face_height = min(face_height, height - y)
        if face_width <= 0 or face_height <= 0:
            continue
        face = image_bgr[y : y + face_height, x : x + face_width]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        crops.append(cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        boxes.append((x, y, face_width, face_height))

    if not crops:
        return image_bgr, []

    probabilities = model.predict(np.asarray(crops), verbose=0).ravel()
    results = []
    annotated = image_bgr.copy()
    for (x, y, face_width, face_height), probability in zip(boxes, probabilities):
        label_index = int(probability >= 0.5)
        label = CLASS_NAMES[label_index]
        confidence = float(probability if label_index else 1.0 - probability)
        text = f"{label}: {confidence * 100:.1f}%"
        color = (40, 190, 90) if label_index else (220, 130, 40)
        cv2.rectangle(annotated, (x, y), (x + face_width, y + face_height), color, 2)
        cv2.putText(
            annotated,
            text,
            (x, max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        results.append(
            {
                "Face": len(results) + 1,
                "Dataset label": label,
                "Model score": f"{confidence * 100:.1f}%",
            }
        )
    return annotated, results


def decode_image(image_bytes):
    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)


def display_prediction(image_bytes, classifier, caption):
    image_bgr = decode_image(image_bytes)
    if image_bgr is None:
        st.error("The image could not be read. Try a JPG or PNG file.")
        return
    annotated, results = classify_faces(image_bgr, classifier, load_face_detector())
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=caption)
    if results:
        st.dataframe(results, hide_index=True, use_container_width=True)
    else:
        st.info("No face was detected. Try a clearer, front-facing image.")


st.title("Face Label Classifier")
st.write("Capture a webcam frame or upload an image to classify each detected face.")
st.warning(
    "The model predicts labels learned from this dataset. It does not determine a person's gender identity."
)

try:
    classifier = load_classifier()
    face_detector = load_face_detector()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

webcam_tab, upload_tab = st.tabs(["Webcam", "Upload image"])
with webcam_tab:
    camera_frame = st.camera_input("Allow camera access, then capture a frame")
    if camera_frame is None:
        st.info("Use the camera control above to take a picture.")
    else:
        display_prediction(camera_frame.getvalue(), classifier, "Annotated webcam capture")

with upload_tab:
    uploaded_image = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG image containing one or more faces.",
    )
    if uploaded_image is None:
        st.info("Choose an image to classify it.")
    else:
        display_prediction(uploaded_image.getvalue(), classifier, "Annotated uploaded image")

st.caption("Educational demonstration. Results can be affected by lighting, pose, occlusion, and dataset bias.")

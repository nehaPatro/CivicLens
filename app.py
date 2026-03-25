import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="Pothole Detection", layout="wide")

st.title("🛣️ AI Pothole Detection System")

# Load YOLOv8 model (auto-downloads)
model = YOLO("yolov8n.pt")

option = st.radio("Select Input Type", ["Image", "Video"])

# ================= IMAGE =================
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            img_array = np.array(image)

            results = model(img_array)

            result_img = results[0].plot()

            st.image(result_img, caption="Detection Result", use_column_width=True)

            st.success(f"Objects detected: {len(results[0].boxes)}")

# ================= VIDEO =================
elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)

        if st.button("Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results[0].plot()

                stframe.image(annotated_frame, channels="BGR")

            cap.release()
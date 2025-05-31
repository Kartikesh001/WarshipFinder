import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Load model
model = YOLO("runs/detect/train11/weights/best.pt")

st.title("🚢 Military Equipment Locator")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

        # Show image with detection
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Image", use_column_width=True)

        # Extract box results
        boxes = results[0].boxes
        if boxes is None or boxes.cls.shape[0] == 0:
            st.warning("⚠️ No objects detected.")
        else:
            st.subheader("📦 Detected Military Objects")
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            class_names = [model.names[c] for c in class_ids]

            for name, conf in zip(class_names, confidences):
                st.write(f"🔹 **{name}** — Confidence: {conf:.2f}")

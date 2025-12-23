# ================================
# STREAMLIT APP - UAP ML
# Klasifikasi Jenis Sampah
# ================================

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Klasifikasi Jenis Sampah",
    layout="centered"
)

st.title("Klasifikasi Jenis Spesifik Sampah")
st.write(
    "Aplikasi klasifikasi citra sampah menggunakan "
    "CNN dan Transfer Learning (ResNet50 & MobileNetV2)"
)

# ================================
# LOAD MODEL, CLASS & EVALUASI
# ================================
@st.cache_resource
def load_resources():
    cnn_model = tf.keras.models.load_model("models/cnn_model.keras")
    resnet_model = tf.keras.models.load_model("models/resnet_model.keras")
    mobilenet_model = tf.keras.models.load_model("models/mobilenet_model.keras")

    class_names = np.load("models/class_names.npy", allow_pickle=True)

    with open("models/evaluation.json") as f:
        evaluation = json.load(f)

    return cnn_model, resnet_model, mobilenet_model, class_names, evaluation


cnn_model, resnet_model, mobilenet_model, class_names, evaluation = load_resources()

# ================================
# PILIH MODEL
# ================================
st.subheader("🔍 Pilih Model Klasifikasi")

model_option = st.selectbox(
    "Model yang digunakan:",
    ("CNN Non-Pretrained", "ResNet50", "MobileNetV2")
)

if model_option == "CNN Non-Pretrained":
    model = cnn_model
    eval_data = evaluation["CNN"]
elif model_option == "ResNet50":
    model = resnet_model
    eval_data = evaluation["ResNet50"]
else:
    model = mobilenet_model
    eval_data = evaluation["MobileNetV2"]

# ================================
# TAMPILKAN HASIL EVALUASI MODEL
# ================================
st.subheader("📊 Evaluasi Model (Data Uji)")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{eval_data['accuracy']*100:.2f}%")
    st.metric("Precision", f"{eval_data['precision']*100:.2f}%")

with col2:
    st.metric("Recall", f"{eval_data['recall']*100:.2f}%")
    st.metric("F1-Score", f"{eval_data['f1_score']*100:.2f}%")

st.info(
    f"Model **{model_option}** dievaluasi menggunakan data uji "
    f"dengan akurasi **{eval_data['accuracy']*100:.2f}%**."
)

st.divider()

# ================================
# UPLOAD GAMBAR
# ================================
st.subheader("📤 Upload Gambar Sampah")

uploaded_file = st.file_uploader(
    "Unggah gambar (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

# ================================
# PROSES & PREDIKSI
# ================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔮 Prediksi Jenis Sampah"):
        with st.spinner("Melakukan prediksi..."):
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        st.success(
            f"### 🧾 Hasil Prediksi: **{class_names[class_index]}**"
        )
        st.write(
            f"Tingkat Kepercayaan Model: **{confidence:.2f}%**"
        )

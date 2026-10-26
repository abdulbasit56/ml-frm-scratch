import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path
import tensorflow as tf
from xgboost import XGBClassifier

from fusion import fuse


BASE_DIR   = Path(__file__).resolve().parent
IMAGE_SIZE = (64, 64)

# must match the column order the XGBoost model was trained on (splits/X_tab_train.csv)
FEATURE_COLS = ['smoke', 'age', 'gender', 'skin_cancer_history', 'fitspatrick', 'itch', 'grew', 'hurt',
                'changed', 'bleed', 'elevation', 'diameter', 'region_ABDOMEN', 'region_ARM', 'region_BACK',
                'region_CHEST', 'region_EAR', 'region_FACE', 'region_FOOT', 'region_FOREARM', 'region_HAND',
                'region_LIP', 'region_NECK', 'region_NOSE', 'region_SCALP', 'region_THIGH']
REGIONS = [c.replace('region_', '') for c in FEATURE_COLS if c.startswith('region_')]

# "Unknown" -> NaN, XGBoost handles missing values the same way it did in training
YES_NO = {"Unknown": np.nan, "No": 0, "Yes": 1}


st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺", layout="centered")
st.title("Skin Cancer Detector")
st.caption("XGBoost on patient metadata + ANN on the lesion image, fused into one prediction")
st.warning("Learning project only, not a medical tool. See a dermatologist about any lesion you're worried about.")


@st.cache_resource
def get_xgb_model():
    model = XGBClassifier()
    model.load_model(BASE_DIR / "models" / "xgb_model.json")
    return model

@st.cache_resource
def get_ann_model():
    return tf.keras.models.load_model(BASE_DIR / "models" / "ann_model.keras")

@st.cache_resource
def get_fusion_params():
    weight    = float(np.load(BASE_DIR / "splits" / "fusion_weight.npy"))
    threshold = float(np.load(BASE_DIR / "splits" / "fusion_threshold.npy"))
    return weight, threshold


def process_image(uploaded_file):
    # same steps as img_processing.py: greyscale -> 64x64 -> scale to 0-1 -> flatten
    img = Image.open(uploaded_file).convert('L')
    img = img.resize(IMAGE_SIZE)
    to_array_img = np.array(img, dtype=np.float32) / 255.0
    return to_array_img.reshape(1, -1)


def build_tabular_row(inputs, region):
    row = {col: 0 for col in FEATURE_COLS}
    row.update(inputs)
    row[f"region_{region}"] = 1
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ── INPUTS ───────────────────────────────────────────────────────────────────
st.subheader("1. Lesion image")
uploaded_file = st.file_uploader("Upload a photo of the lesion", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, width=250)

st.subheader("2. Patient details")
col1, col2 = st.columns(2)
with col1:
    age         = st.number_input("Age", min_value=1, max_value=110, value=50)
    gender      = st.selectbox("Gender", options=["Unknown", "Female", "Male"])
    fitspatrick = st.selectbox("Fitzpatrick skin type", options=["Unknown", 1, 2, 3, 4, 5, 6])
    region      = st.selectbox("Body region", options=REGIONS, format_func=lambda x: x.title())
    diameter    = st.number_input("Lesion diameter (mm)", min_value=0.0, max_value=100.0, value=None,
                                  step=1.0, placeholder="Unknown")
    smoke       = st.selectbox("Smokes", options=list(YES_NO))
    history     = st.selectbox("Previous skin cancer", options=list(YES_NO))
with col2:
    itch      = st.selectbox("Itches", options=list(YES_NO))
    grew      = st.selectbox("Has grown", options=list(YES_NO))
    hurt      = st.selectbox("Hurts", options=list(YES_NO))
    changed   = st.selectbox("Has changed", options=list(YES_NO))
    bleed     = st.selectbox("Bleeds", options=list(YES_NO))
    elevation = st.selectbox("Raised / elevated", options=list(YES_NO))


# ── PREDICTION ───────────────────────────────────────────────────────────────
if st.button("Predict", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a lesion image first.")
    else:
        try:
            xgb_model         = get_xgb_model()
            ann_model         = get_ann_model()
            weight, threshold = get_fusion_params()

            inputs = {
                'smoke':               YES_NO[smoke],
                'age':                 age,
                'gender':              {"Unknown": np.nan, "Female": 0, "Male": 1}[gender],
                'skin_cancer_history': YES_NO[history],
                'fitspatrick':         np.nan if fitspatrick == "Unknown" else fitspatrick,
                'itch':                YES_NO[itch],
                'grew':                YES_NO[grew],
                'hurt':                YES_NO[hurt],
                'changed':             YES_NO[changed],
                'bleed':               YES_NO[bleed],
                'elevation':           YES_NO[elevation],
                'diameter':            np.nan if diameter is None else diameter,
            }
            X_tab = build_tabular_row(inputs, region)
            X_img = process_image(uploaded_file)

            tree_prob  = float(xgb_model.predict_proba(X_tab)[0, 1])
            ann_prob   = float(ann_model.predict(X_img, verbose=0).flatten()[0])
            fused_prob = float(fuse(tree_prob, ann_prob, weight))

            m1, m2, m3 = st.columns(3)
            m1.metric("Metadata model", f"{tree_prob:.1%}")
            m2.metric("Image model", f"{ann_prob:.1%}")
            m3.metric("Fused", f"{fused_prob:.1%}")

            if fused_prob > threshold:
                st.error(f"Flagged as possibly MALIGNANT (fused probability above the {threshold:.0%} threshold).")
            else:
                st.success(f"Likely BENIGN (fused probability below the {threshold:.0%} threshold).")

            st.caption(f"Fusion: {weight:.0%} metadata model + {1 - weight:.0%} image model. The threshold is set low "
                       f"on purpose so fewer malignant cases are missed, so expect some false alarms.")

        except FileNotFoundError:
            st.error("Saved models not found. Run `xgb.py`, `ANN.py` and then `fusion.py` first.")

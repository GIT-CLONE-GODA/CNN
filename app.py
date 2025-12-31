import streamlit as st
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# --- استيراد الموديلات ---
try:
    from models import AttentionUNet, TransUNetSkip
except ImportError:
    st.error("Error: 'model_definitions.py' not found.")

# 1. إعدادات الصفحة
st.set_page_config(page_title="Skin Lesion Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔬 Skin Lesion Analysis Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# 2. الجانب الأيسر
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Architecture", ("Attention U-Net", "TransUNet Skip"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. تحميل الموديل
@st.cache_resource
def load_model(m_type):
    if m_type == "Attention U-Net":
        model = AttentionUNet().to(device)
        path = "best_attention_unet.pth" 
    else:
        model = TransUNetSkip().to(device)
        path = "best_transunet_skip.pth"
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    return model

# 4. التحويلات (من كودك)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 5. رفع الصورة
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner('Processing...'):
        model = load_model(model_type)
        input_tensor = val_transform(image=image_np)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            # عتبة التثبيت 0.5 برمجياً
            pred_mask = (output[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    # تجهيز الصور للعرض
    # أ- الماسك (أبيض وأسود)
    mask_resized = np.array(Image.fromarray(pred_mask).resize(image.size, resample=Image.NEAREST))
    mask_pil = Image.fromarray(mask_resized * 255)

    # ب- الـ Overlay (دمج شفاف)
    overlay_np = image_np.copy()
    overlay_np[mask_resized == 1] = [255, 0, 0] # تلوين منطقة الورم بالأحمر
    final_overlay = Image.blend(image, Image.fromarray(overlay_np), alpha=0.4)

    # --- 6. العرض في 3 أعمدة جنباً إلى جنب ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📷 Original")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🎭 Binary Mask")
        st.image(mask_pil, use_container_width=True)

    with col3:
        st.markdown("### 🎯 Overlay Result")
        st.image(final_overlay, use_container_width=True)
    
    st.success(f"Successfully analyzed using {model_type}")
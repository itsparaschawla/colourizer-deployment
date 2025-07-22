# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# from io import BytesIO
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

# # === Preprocessing Function ===
# def preprocess_image(uploaded_image):
#     img_rgb = np.array(uploaded_image.convert("RGB"))
#     img_rgb = cv2.resize(img_rgb, (128, 128))
#     img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

#     L_channel = img_lab[:, :, 0:1].astype(np.float32) / 255.0
#     ab_channels = (img_lab[:, :, 1:].astype(np.float32) - 128.0) / 127.0
#     return L_channel, ab_channels, img_rgb  # Return original resized RGB too

# # === Lab to RGB Conversion ===
# def lab_to_rgb(L, ab):
#     L = L * 100.0
#     ab = ab * 128.0
#     lab = np.concatenate((L, ab), axis=-1).astype(np.float32)
#     rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
#     rgb = np.clip(rgb, 0, 1)
#     return rgb

# # === Custom Metrics for Loading Model ===
# @tf.keras.utils.register_keras_serializable()
# def ssim_metric(y_true, y_pred):
#     y_true_scaled = (y_true + 1) / 2
#     y_pred_scaled = (y_pred + 1) / 2
#     return tf.reduce_mean(tf.image.ssim(y_true_scaled, y_pred_scaled, max_val=1.0))

# @tf.keras.utils.register_keras_serializable()
# def psnr_metric(y_true, y_pred):
#     return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=2.0))

# # === Load Model ===
# @st.cache_resource
# def load_colorization_model():
#     return tf.keras.models.load_model("mountains_forest_u_net_best.h5", custom_objects={
#         'ssim_metric': ssim_metric,
#         'psnr_metric': psnr_metric,
#         'mse': tf.keras.losses.MeanSquaredError()  # <- This is the fix
#     })

# model = load_colorization_model()

# # === Streamlit UI ===
# st.title("B/W Image Colorization 🌄🌲(Mountains & Forest)")
# st.markdown("Upload a grayscale image of a mountain or forest landscape and get the colorized version with PSNR/SSIM metrics.")

# uploaded_file = st.sidebar.file_uploader("Upload a black & white image of a mountain or forest landscape", type=["jpg", "png", "jpeg"])

# # if uploaded_file:
# #     uploaded_image = Image.open(uploaded_file)
# #     st.image(uploaded_image, caption="Original Image", use_container_width=True)

# #     with st.spinner("Colorizing..."):
# #         L_channel, ab_true, original_resized = preprocess_image(uploaded_image)
# #         input_L = np.expand_dims(L_channel, axis=0)  # (1, 128, 128, 1)

# #         predicted_ab = model.predict(input_L)[0]  # (128, 128, 2)
# #         colorized_image = lab_to_rgb(L_channel, predicted_ab)  # [0, 1] float

# #         # Convert colorized image to uint8
# #         colorized_uint8 = (colorized_image * 255).astype(np.uint8)

# #         # Show colorized image
# #         st.image(colorized_uint8, caption="Colorized Output", use_container_width=True)

# if uploaded_file:
#     uploaded_image = Image.open(uploaded_file)

#     with st.spinner("Colorizing..."):
#         L_channel, ab_true, original_resized = preprocess_image(uploaded_image)
#         input_L = np.expand_dims(L_channel, axis=0)  # (1, 128, 128, 1)

#         predicted_ab = model.predict(input_L)[0]  # (128, 128, 2)
#         colorized_image = lab_to_rgb(L_channel, predicted_ab)  # [0, 1] float

#         # Convert colorized image to uint8
#         colorized_uint8 = (colorized_image * 255).astype(np.uint8)

#     # Display side-by-side images
#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(uploaded_image, caption="Original Image", use_container_width=True)

#     with col2:
#         st.image(colorized_uint8, caption="Colorized Output", use_container_width=True)


#         # Compute metrics using original resized image
#         true_rgb = original_resized.astype(np.float32) / 255.0
#         ssim_value = ssim(true_rgb, colorized_image, channel_axis=-1, data_range=1.0)
#         psnr_value = psnr(true_rgb, colorized_image, data_range=1.0)

#         st.markdown(f"**SSIM**: `{ssim_value:.4f}`  |  **PSNR**: `{psnr_value:.2f} dB`")

#         # Download Button
#         result_pil = Image.fromarray(colorized_uint8)
#         buffer = BytesIO()
#         result_pil.save(buffer, format="PNG")
#         st.download_button(
#             label="📥 Download Colorized Image",
#             data=buffer.getvalue(),
#             file_name="colorized_output.png",
#             mime="image/png"
#         )


import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# Set your sample images directory (relative to app.py)
SAMPLE_DIR = "test_samples"

# Build the matcher between BW and color filenames
def get_sample_lists():
    # B&W images: testN.jpg
    bw_files = [f for f in os.listdir(SAMPLE_DIR) if f.startswith("test") and f.endswith(".jpg")]
    # Map: testN.jpg => originalN.jpg
    color_map = {}
    for bw in bw_files:
        idx = bw.replace("test", "").replace(".jpg","")
        col = f"original{idx}.jpg"
        if os.path.exists(os.path.join(SAMPLE_DIR, col)):
            color_map[bw] = col
        else:
            color_map[bw] = None
    return bw_files, color_map

bw_files, color_file_map = get_sample_lists()

def preprocess_image(uploaded_image):
    img_rgb = np.array(uploaded_image.convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (128, 128))
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L_channel = img_lab[:, :, 0:1].astype(np.float32) / 255.0
    ab_channels = (img_lab[:, :, 1:].astype(np.float32) - 128.0) / 127.0
    return L_channel, ab_channels, img_rgb

def lab_to_rgb(L, ab):
    L = L * 100.0
    ab = ab * 128.0
    lab = np.concatenate((L, ab), axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb, 0, 1)
    return rgb

@tf.keras.utils.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    y_true_scaled = (y_true + 1) / 2
    y_pred_scaled = (y_pred + 1) / 2
    return tf.reduce_mean(tf.image.ssim(y_true_scaled, y_pred_scaled, max_val=1.0))

@tf.keras.utils.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=2.0))

@st.cache_resource
def load_colorization_model():
    return tf.keras.models.load_model(
        "mountains_forest_u_net_best.h5",
        custom_objects={'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric, 'mse': tf.keras.losses.MeanSquaredError()}
    )
model = load_colorization_model()

st.title("B/W Image Colorization 🌄🌲 (Mountains & Forest)")
st.markdown("Upload a grayscale mountain/forest image or select a sample test image. You’ll see the original color image, input grayscale, and colorized output with PSNR/SSIM metrics.")

# === UI: Sidebar File Uploader & Sample Selector ===
st.sidebar.header("Try it out:")

uploaded_file = st.sidebar.file_uploader(
    "Upload a black & white image (mountain or forest)", type=["jpg", "png", "jpeg"]
)
st.sidebar.markdown("**Or choose a sample image:**")

# Option: None or available testN.jpg
sample_choice = st.sidebar.selectbox("Sample Test Images", ["None"] + bw_files)

if sample_choice != "None":
    bw_path = os.path.join(SAMPLE_DIR, sample_choice)
    uploaded_image = Image.open(bw_path)
    color_image = None
    color_name = color_file_map.get(sample_choice, None)
    if color_name and os.path.exists(os.path.join(SAMPLE_DIR, color_name)):
        color_image = Image.open(os.path.join(SAMPLE_DIR, color_name))
elif uploaded_file:
    uploaded_image = Image.open(uploaded_file)
    color_image = None
else:
    uploaded_image = None
    color_image = None

if uploaded_image:
    with st.spinner("Colorizing..."):
        L_channel, ab_true, original_resized = preprocess_image(uploaded_image)
        input_L = np.expand_dims(L_channel, axis=0)
        predicted_ab = model.predict(input_L)[0]
        colorized_image = lab_to_rgb(L_channel, predicted_ab)
        colorized_uint8 = (colorized_image * 255).astype(np.uint8)

        # Prepare three views
        input_gray_uint8 = (L_channel.squeeze() * 255).astype(np.uint8)
        input_gray_3ch = np.stack([input_gray_uint8]*3, axis=-1)
        orig_image_display = color_image if color_image is not None else uploaded_image

    # === Display Images Side-by-Side ===
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(orig_image_display, caption="Original Color", use_container_width=True)
    with col2:
        st.image(input_gray_3ch, caption="Input (B&W)", use_container_width=True)
    with col3:
        st.image(colorized_uint8, caption="Colorized Output", use_container_width=True)

    # === Metrics (only when color ground truth present) ===
    if color_image is not None:
        true_rgb = np.array(color_image.resize((128,128))).astype(np.float32) / 255.0
        ssim_value = ssim(true_rgb, colorized_image, channel_axis=-1, data_range=1.0)
        psnr_value = psnr(true_rgb, colorized_image, data_range=1.0)
        st.markdown(f"**SSIM:** `{ssim_value:.4f}`  |  **PSNR:** `{psnr_value:.2f} dB`")
    else:
        st.info("Load a sample with original color to see SSIM/PSNR metrics.")

    # === Download Result ===
    result_pil = Image.fromarray(colorized_uint8)
    buffer = BytesIO()
    result_pil.save(buffer, format="PNG")
    st.download_button(
        label="📥 Download Colorized Image",
        data=buffer.getvalue(),
        file_name="colorized_output.png",
        mime="image/png"
    )

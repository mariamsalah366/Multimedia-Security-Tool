import streamlit as st
import cv2
import numpy as np
import tempfile
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

# Initialize the session state to track the current page
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to navigate back to home page
def go_home():
    st.session_state.page = 'home'

# Function to add watermark text in a specific position
def add_watermark_text(img, watermark_text, position="Center"):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    width, height = img.size

    #textbbox
    text_bbox = draw.textbbox((0, 0), watermark_text, font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]


    if position == "Top-Left":
        pos = (10, 10)
    elif position == "Top-Right":
        pos = (width - text_width - 10, 10)
    elif position == "Bottom-Left":
        pos = (10, height - text_height - 10)
    elif position == "Bottom-Right":
        pos = (width - text_width - 10, height - text_height - 10)
    elif position == "Center":
        pos = ((width - text_width) // 2, (height - text_height) // 2)

    draw.text(pos, watermark_text, font=font, fill=(255, 255, 255, 128))
    return img

def embed_text_into_image(img, secret_text):
    img_arr = np.array(img)  # Keep the original image (no grayscale conversion)
    flat_img = img_arr.flatten()

    secret_text += "####"  # Delimiter
    binary_secret = ''.join(format(ord(c), '08b') for c in secret_text)

    if len(binary_secret) > len(flat_img):
        raise ValueError("Secret message too large for the image!")

    for i in range(len(binary_secret)):
        flat_img[i] = (flat_img[i] & 0xFE) | int(binary_secret[i])

    stego_img_arr = flat_img.reshape(img_arr.shape)
    stego_img = Image.fromarray(stego_img_arr.astype('uint8'))  # Keep original color channels
    return stego_img

# Function to apply watermark on video
def apply_watermark(input_video_path, output_video_path):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    watermark = '1010101010101010'  # Example watermark
    wm_len = len(watermark)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        wm_index = 0
        for y in range(height):
            for x in range(width):
                if wm_index >= wm_len:
                    break
                frame[y, x, 0] = (frame[y, x, 0] & 254) | int(watermark[wm_index])
                wm_index += 1
            if wm_index >= wm_len:
                break

        output.write(frame)

    video.release()
    output.release()

    print("Watermarking done!")
    print(f"Watermarked video saved as: {output_video_path}")

    return output_video_path

##################### Home Page #####################
if st.session_state.page == 'home':
    st.title("🏠 Multimedia Security Project")

    st.markdown("---")
    st.subheader("Please Select a Section:")
    st.markdown(" ")

    # Create three columns 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image.png", width=150)  
        if st.button("🔹 Image Processing"):
            st.session_state.page = 'image_processing'

    with col2:
        st.image("watermark.png", width=150)  
        if st.button("🔹 Watermarking"):
            st.session_state.page = 'watermarking'

    with col3:
        st.image("encrypted.png", width=150)  
        if st.button("🔹 Steganography"):
            st.session_state.page = 'steganography'

    st.markdown("---")
    st.markdown(
        """
        **Developed by:**
        
        - Mariam Salah  
        - Fatma Hassan
        - Malak Walid
        - Adham Hitham 
        - Ammar Ashraf
        """
    )

##################### Image Processing Page #####################
elif st.session_state.page == 'image_processing':
    st.title("🖼️ Image Processing Section")

    if st.button("⬅️ Back To Home"):
        go_home()

    st.write("Upload an Image Here To Perform Processing Operations.")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.markdown("---")
        st.subheader("✨ Choose Processing Operation:")

        operation = st.radio(
            "Select Operation:",
            ("None", "✂️ Crop", "📏 Resize", "🔄 Rotate", "📝 Text Watermark"),
            horizontal=True
        )

        if operation == "✂️ Crop":
            st.subheader("✂️ Crop Image")
            x1 = st.number_input("Start X", min_value=0, value=0)
            y1 = st.number_input("Start Y", min_value=0, value=0)
            x2 = st.number_input("End X", min_value=0, value=img.width)
            y2 = st.number_input("End Y", min_value=0, value=img.height)

            if st.button("Crop Now"):
                cropped_img = img.crop((x1, y1, x2, y2))
                st.image(cropped_img, caption="Cropped Image", use_container_width=True)

        elif operation == "📏 Resize":
            st.subheader("📏 Resize Image")
            new_width = st.number_input("New Width", min_value=1, value=img.width)
            new_height = st.number_input("New Height", min_value=1, value=img.height)

            if st.button("Resize Now"):
                resized_img = img.resize((new_width, new_height))
                st.image(resized_img, caption="Resized Image", use_container_width=True)

        elif operation == "🔄 Rotate":
            st.subheader("🔄 Rotate Image")
            angle = st.number_input("Angle", min_value=-360, max_value=360, value=0)

            if st.button("Rotate Now"):
                rotated_img = img.rotate(angle, expand=True)
                st.image(rotated_img, caption="Rotated Image", use_container_width=True)

        elif operation == "📝 Text Watermark":
            st.subheader("📝 Text Watermark")
            watermark_text = st.text_input("Enter Watermark Text", value="Sample Watermark")

            position = st.selectbox("Choose Position", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"])

            if st.button("Apply Watermark"):
                img_with_watermark = add_watermark_text(img, watermark_text, position)
                st.image(img_with_watermark, caption="Watermarked Image", use_container_width=True)



##################### Watermarking Main Page #####################
elif st.session_state.page == 'watermarking':
    st.title("🔹 Watermarking Section")

    if st.button("⬅️ Back To Home"):
        go_home()

    st.write("Choose The Type Of Watermarking:")

    if st.button("🖼️ Image Watermarking"):
        st.session_state.page = 'image_watermarking'

    if st.button("🎥 Video Watermarking"):
        st.session_state.page = 'video_watermarking'

    if st.button("🎵 Audio Watermarking"):
        st.session_state.page = 'audio_watermarking'




##################### Image Watermarking Page #####################
elif st.session_state.page == 'image_watermarking':
    st.title("🖼️ Image Watermarking")

    if st.button("⬅️ Back to Watermarking"):
        st.session_state.page = 'watermarking'

    st.write("Upload An Image Here And Apply Watermarking Operations.")

    # Uploading the image and watermark
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    uploaded_watermark = st.file_uploader("Upload a Watermark Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None and uploaded_watermark is not None:
        img = Image.open(uploaded_image).convert('RGB')
        watermark = Image.open(uploaded_watermark).convert('RGB')

        img = img.resize((600, 600)) 
        watermark = watermark.resize((600, 600))

        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.image(watermark, caption="Uploaded Watermark", use_container_width=True)

        # Adding Radio button for selecting watermarking method
        watermark_method = st.radio(
            "Select Watermarking Technique:",
            ("Overlaying", "Other Techniques")
        )

        if watermark_method == "Overlaying":
            st.subheader("✨ Overlaying Methods:")
            watermark_overlay_method = st.selectbox(
                "Select Overlaying Method", 
                ("None", "Additive", "Transparency", "Multiplicative")
            )

            if watermark_overlay_method == "Additive":
                k = 0.3
                watermarked = np.clip(np.array(img) + k * np.array(watermark), 0, 255).astype(np.uint8)
                result = Image.fromarray(watermarked)
                st.image(result, caption="Additive Watermarked Image", use_container_width=True)

            elif watermark_overlay_method == "Transparency":
                k = 0.3
                transparent = (1 - k) * np.array(img) + k * np.array(watermark)
                transparent = np.clip(transparent, 0, 255).astype(np.uint8)
                result = Image.fromarray(transparent)
                st.image(result, caption="Transparency Watermarked Image", use_container_width=True)

            elif watermark_overlay_method == "Multiplicative":
                k = 0.3
                watermarked = np.clip(np.array(img) * (1 + k * np.array(watermark) / 255), 0, 255).astype(np.uint8)
                result = Image.fromarray(watermarked)
                st.image(result, caption="Multiplicative Watermarked Image", use_container_width=True)

        elif watermark_method == "Other Techniques":
            st.subheader("✨ Other Methods:")

            method = st.selectbox(
            "Choose a Watermarking Technique:",
            ["None", "🔸 DCT Watermarking", "🔸 DWT Watermarking", "🔸 HSI Watermarking"])

            if method == "🔸 DCT Watermarking":
                img_dct = cv2.dct(np.float32(np.array(img.convert('L'))))
                watermark_dct = cv2.dct(np.float32(np.array(watermark.convert('L'))))
                dct_watermarked = img_dct + 0.3 * watermark_dct
                dct_watermarked = cv2.idct(dct_watermarked).clip(0, 255).astype(np.uint8)
                result = Image.fromarray(dct_watermarked)
                st.image(result, caption="DCT Watermarked Image", use_container_width=True)

            elif method == "🔸 DWT Watermarking":
                img_gray = np.array(img.convert('L'))
                watermark_gray = np.array(watermark.convert('L'))
                coeffs2 = pywt.dwt2(img_gray, 'haar')
                LL, (LH, HL, HH) = coeffs2
                watermark_gray_resized = cv2.resize(watermark_gray, (LL.shape[1], LL.shape[0]))
                coeffs2_watermarked = (LL + 0.3 * watermark_gray_resized, (LH, HL, HH))
                img_dwt = pywt.idwt2(coeffs2_watermarked, 'haar')
                img_dwt = np.clip(img_dwt, 0, 255).astype(np.uint8)
                result = Image.fromarray(img_dwt)
                st.image(result, caption="DWT Watermarked Image", use_container_width=True)

            elif method == "🔸 HSI Watermarking":
                img_hsi = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
                watermark_hsi = cv2.cvtColor(np.array(watermark), cv2.COLOR_RGB2HSV)
                img_hsi[..., 2] = np.clip(img_hsi[..., 2] + 0.3 * watermark_hsi[..., 2], 0, 255)
                result_hsi = cv2.cvtColor(img_hsi, cv2.COLOR_HSV2RGB)
                result = Image.fromarray(result_hsi)
                st.image(result, caption="HSI Watermarked Image", use_container_width=True)
                

##################### Video Watermarking Page #####################
elif st.session_state.page == 'video_watermarking':
    st.title("🎥 Video Watermarking")

    if st.button("⬅️ Back to Watermarking"):
        st.session_state.page = 'watermarking'

    st.write("Upload a video here and apply watermarking operations.")

    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input_video:
            tmp_input_video.write(uploaded_video.read())
            tmp_input_video_path = tmp_input_video.name

        # Generate the output file path
        with tempfile.NamedTemporaryFile(delete=False, suffix="_watermarked.mp4") as tmp_output_video:
            output_video_path = tmp_output_video.name

        if st.button("Apply Watermark"):
            # Apply watermarking and save the watermarked video
            apply_watermark(tmp_input_video_path, output_video_path)

            # Provide a download link for the watermarked video
            with open(output_video_path, 'rb') as f:
                st.download_button(
                    label="Download Watermarked Video",
                    data=f,
                    file_name="watermarked_video.mp4",
                    mime="video/mp4"
                )


##################### Audio Watermarking Page #####################
elif st.session_state.page == 'audio_watermarking':
    st.title("🎵 Audio Watermarking")

    if st.button("⬅️ Back to Watermarking"):
        st.session_state.page = 'watermarking'

    st.write("Upload an audio file here and apply watermarking operations.")


##################### Steganography Page #####################
elif st.session_state.page == 'steganography':
    st.title("🔹 Steganography Section")
    if st.button("⬅️ Back to Home"):
        go_home()

    st.write("Upload an Image to Embed a Hidden Message")

    uploaded_image = st.file_uploader("Upload Image for Steganography", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        secret_text = st.text_area("Enter the Secret Text")
        if st.button("Embed Secret Text"):
            stego_img = embed_text_into_image(img, secret_text)
            st.image(stego_img, caption="Image with Embedded Message", use_container_width=True)
            # Save image in session state for later use if needed
            st.session_state.stego_img = stego_img


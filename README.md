# Multimedia Security Project

## Overview
This application provides a comprehensive set of tools for multimedia security operations, including image processing, watermarking, and steganography. Built using Streamlit, the application offers an intuitive interface for performing various security and manipulation operations on different media types (images, videos, and audio).

## Features

### 1. Image Processing
- **Crop**: Crop images by specifying coordinates
- **Resize**: Resize images to desired dimensions
- **Rotate**: Rotate images by specified angles
- **Text Watermark**: Add text watermarks to images with position control

### 2. Watermarking
The application supports watermarking for different media types:

#### Image Watermarking
- **Visible Watermarks**: Add visible watermarks to images
- **Invisible Watermarks**: Apply various techniques for invisible watermarking:
  - LSB (Least Significant Bit)
  - DCT (Discrete Cosine Transform)
  - DWT (Discrete Wavelet Transform)

#### Video Watermarking
- **Least Significant Bit (LSB)**: Embed watermark data in the least significant bits of video frames
- **DCT-based Watermarking**: Apply DCT-based watermarking to video frames

#### Audio Watermarking
- **LSB-based Audio Watermarking**: Embed watermark in the least significant bits of audio samples
- **DCT-based Audio Watermarking**: Use discrete cosine transform for audio watermarking

### 3. Steganography
- **Image Steganography**: Hide text messages within images using LSB technique
- **Video Steganography**: Embed text messages within video files
- **Audio Steganography**: Hide text within audio files

## Technical Implementation

### Image Processing Functions
- `add_watermark_text()`: Adds text watermark to an image at a specified position
- Image manipulation using PIL (Python Imaging Library)

### Watermarking Techniques
- **LSB-based Watermarking**: 
  - Modifies the least significant bit of pixel values or audio samples
  - Implementation in `embed_text_into_image()` for images and `apply_watermark_lsb()` for videos
  
- **DCT-based Watermarking**:
  - Uses discrete cosine transform to embed watermark in frequency domain
  - Implementation in `apply_watermark_dct()` for videos and `embed_dct()` for audio
  
- **DWT-based Watermarking**:
  - Uses discrete wavelet transform for robust watermarking
  - Implemented using PyWavelets library

### Steganography Implementation
- **Image Steganography**: Uses LSB modification to embed text (`embed_text_into_image()`)
- **Video Steganography**: Embeds text within video frames (`embed_text_in_video_cv2()`)
- **Audio Steganography**: Hides text within audio samples

### Utility Functions
- `audio_to_wav()`: Converts audio files to WAV format for processing
- `extract_audio()`: Extracts audio from video files
- `text_to_binary()`: Converts text to binary representation for embedding

## User Interface
The application uses Streamlit for a clean, interactive user interface:
- **Home Page**: Navigation to different sections
- **Separate Sections**: For image processing, watermarking, and steganography
- **Interactive Elements**: File uploaders, sliders, input fields, and buttons
- **Results Display**: Shows processed media with download options

## Dependencies
- streamlit: For the web interface
- PIL/Pillow: For image processing
- OpenCV (cv2): For video processing
- numpy: For numerical operations
- pywt: For wavelet transforms
- scipy: For signal processing operations
- pydub: For audio processing
- moviepy: For video editing and extraction

## Usage
1. Run the application using: `streamlit run app.py`
2. Navigate to the desired section (Image Processing, Watermarking, or Steganography)
3. Upload the media files
4. Apply the desired operations
5. View and save the results

## Security Considerations
- The watermarking and steganography techniques implemented offer various levels of robustness against attacks
- LSB-based methods offer simplicity but less robustness
- Transform domain methods (DCT, DWT) provide better resistance to manipulation
- The application demonstrates concepts and should be enhanced for production use cases

## Developers
- Mariam Salah
- Fatma Hassan
- Malak Walid
- Adham Hitham
- Ammar Ashraf 
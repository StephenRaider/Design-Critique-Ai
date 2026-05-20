import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from huggingface_hub import InferenceClient

# --- Configuration & UI Setup ---
st.set_page_config(page_title="AI Design Critic", layout="wide")
st.title("🎨 AI Poster Design Critic (VLM Powered)")

# Pull API keys securely from your secrets.toml
try:
    hf_key = st.secrets["HF_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    hf_key = None
    gemini_key = None
    st.error("⚠️ Missing API Keys. Please ensure HF_API_KEY and GEMINI_API_KEY are in .streamlit/secrets.toml")

st.markdown("**The Pipeline:**\n1. OpenCV extracts raw math (Text Density, Peak Vibrancy, Colors).\n2. Gemini 3.1 Flash Lite identifies the Art & Typography Style.\n3. Qwen 2.5 synthesizes both into an Art Director critique.")

# --- 1. Computer Vision Functions (OpenCV) ---

def process_colors(image, k=4):
    blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    image_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(image_rgb, (150, 150), interpolation=cv2.INTER_AREA)
    pixels = small_image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    counts = Counter(labels)
    
    total_pixels = len(pixels)
    colors_info = []
    
    for i in counts.keys():
        color = kmeans.cluster_centers_[i]
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b).upper()
        percentage = round((counts[i] / total_pixels) * 100, 2)
        
        colors_info.append({"rgb": [r, g, b], "hex": hex_code, "percentage": percentage})
        
    colors_info.sort(key=lambda x: x['percentage'], reverse=True)
    if len(colors_info) == 4 and colors_info[3]['percentage'] < 5.0:
        colors_info = colors_info[:3]
        
    return colors_info

def analyze_text_and_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_image = image.copy()
    img_h, img_w = image.shape[:2]
    
    text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 15 and h > 10) and (h < img_h * 0.3) and (w < img_w * 0.9):
            if (w / float(h)) > 1.2: 
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
                
    text_area_pixels = cv2.countNonZero(text_mask)
    total_pixels = img_h * img_w
    text_area_percentage = round((text_area_pixels / total_pixels) * 100, 1)
            
    text_pixels = gray[thresh == 255]
    bg_pixels = gray[thresh == 0]
    mean_text_lum = text_pixels.mean() if len(text_pixels) > 0 else 0
    mean_bg_lum = bg_pixels.mean() if len(bg_pixels) > 0 else 0
    contrast_score = round(abs(mean_text_lum - mean_bg_lum) / 255.0 * 100, 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    vibrancy_score = round(np.percentile(s, 90) / 255.0 * 100, 1)
    
    return annotated_image, text_area_percentage, contrast_score, vibrancy_score


# --- 2. Vision-Language Model (Gemini) ---

def get_vibe_check(opencv_image, api_key):
    """Passes the image to Gemini 3.1 Flash Lite to determine the stylistic 'vibe'."""
    genai.configure(api_key=api_key)
    
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    
    # Prompt updated to strictly request typography classification
    vlm_prompt = (
        "You are an expert graphic design analyst. Look at this poster and describe its "
        "overall aesthetic, specific typography style (e.g., bold sans-serif, elegant serif, handwritten, display), and vibe in 2-3 short sentences. "
        "Mention specific design movements (e.g., 'Japanese Maximalist', 'Swiss Corporate', 'Grunge', 'Minimalist'). "
        "Do not critique the layout, just identify the stylistic intent."
    )
    
    try:
        response = model.generate_content([vlm_prompt, pil_image])
        return response.text
    except Exception as e:
        return f"VLM Error: {str(e)}"


# --- 3. Text LLM Engine (Qwen) ---

def generate_critique(opencv_data, vibe_text, api_key):
    """Sends data + Vibe Check to Qwen 2.5 using the Hugging Face SDK."""
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=api_key)
    
    system_prompt = (
        "You are a Senior Art Director and expert graphic design critic. Your task is to analyze a poster based on objective computer vision metrics AND a stylistic 'Vibe Check' from a Vision Model. "
        "CRITICAL INSTRUCTIONS: Do not blindly follow basic rules. Interpret the OpenCV math through the lens of the Vibe Check. "
        "The Vibe Check will define the typography style—rely entirely on it for font-related critique. "
        "If the Vibe Check says it's a 'Vibrant Anime Ad', do not tell them to lower the saturation. If it says 'Maximalist', do not complain about text density. "
        "Format your response using Markdown with three clear headings: 1. Composition & Hierarchy, 2. Color & Contrast, 3. Art Director's Suggestions."
    )
    
    user_data = (
        f"Art Direction & Typography Vibe (From Vision Model):\n"
        f"{vibe_text}\n\n"
        f"Computer Vision Metrics (From OpenCV):\n"
        f"- Text Density: {opencv_data['text_area_percentage']}% of the total image area is occupied by text bounding boxes.\n"
        f"- Text-to-Background Contrast Score: {opencv_data['contrast']}/100.\n"
        f"- Peak Vibrancy Score (90th Percentile): {opencv_data['vibrancy']}%.\n"
        f"- Dominant Colors: {opencv_data['color_paragraph']}\n\n"
        f"Provide your professional critique based on how these metrics interact with the intended vibe."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_data}
    ]
    
    try:
        response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.3)
        final_output = response.choices[0].message.content
        return str(messages), final_output
    except Exception as e:
        return str(messages), f"API Error: {str(e)}"


# --- 4. Main Application Flow ---

uploaded_file = st.file_uploader("Upload a Poster Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if not hf_key or not gemini_key:
        st.warning("⚠️ Cannot proceed without both API keys.")
        st.stop()
        
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner("Step 1: OpenCV extracting math metrics..."):
        colors = process_colors(image)
        annotated_img, text_area_percentage, contrast, vibrancy = analyze_text_and_metrics(image)
        
        color_strings = [f"{c['hex']} at {c['percentage']}%" for c in colors]
        color_paragraph = ", ".join(color_strings[:-1]) + f", and {color_strings[-1]}" if len(color_strings) > 1 else color_strings[0]
        
        opencv_data = {
            "text_area_percentage": text_area_percentage,
            "contrast": contrast,
            "vibrancy": vibrancy,
            "color_paragraph": color_paragraph
        }
        
    with st.spinner("Step 2: Gemini 3.1 Flash Lite checking visual vibe..."):
        vibe_description = get_vibe_check(image, gemini_key)
        
        if "VLM Error" in vibe_description:
            st.error(f"Gemini failed to analyze the image: {vibe_description}")
            st.stop()
    
    with st.spinner("Step 3: Qwen 2.5 writing Art Director critique..."):
        sent_prompt, llm_response = generate_critique(opencv_data, vibe_description, hf_key)

    # --- Output Dashboard ---
    st.divider()
    
    col_img, col_vibe = st.columns([1, 1])
    with col_img:
        st.subheader("1. Annotated Poster")
        st.image(annotated_img, channels="BGR", caption=f"Text Area: {text_area_percentage}%", use_container_width=True)
    with col_vibe:
        st.subheader("2. VLM Aesthetic Analysis")
        st.info(vibe_description)
        st.markdown("#### Raw OpenCV Data")
        st.json(opencv_data)
        
    st.divider()
    st.subheader("3. Final Art Director Critique (Qwen 2.5)")
    st.success(llm_response)
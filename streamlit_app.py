import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from huggingface_hub import InferenceClient
import json

# --- Configuration & UI Setup ---
st.set_page_config(page_title="AI Design Critic", layout="wide")

# Custom CSS for the Dashboard UI cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        border: 1px solid #333;
    }
    .metric-title {
        font-size: 14px;
        color: #AAAAAA;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 AI Poster Design Critic (VLM Dashboard)")
st.markdown("Upload a design to generate a professional, data-driven critique.")

# Pull API keys securely from your secrets.toml
try:
    hf_key = st.secrets["HF_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    hf_key = None
    gemini_key = None
    st.error("⚠️ Missing API Keys. Please ensure HF_API_KEY and GEMINI_API_KEY are in .streamlit/secrets.toml")

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
        percentage = round((counts[i] / total_pixels) * 100, 1)
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
    text_area_percentage = round((text_area_pixels / (img_h * img_w)) * 100, 1)
            
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    vibrancy_score = round(np.percentile(s, 90) / 255.0 * 100, 1)

    # Temperature Metric (Red vs Blue ratio)
    mean_b = image[:, :, 0].mean()
    mean_r = image[:, :, 2].mean()
    temp_score = round((mean_r / (mean_r + mean_b + 1e-5)) * 100, 1)
    
    return annotated_image, text_area_percentage, vibrancy_score, temp_score


# --- 2. Vision-Language Model (Gemini) ---

def get_vibe_check(opencv_image, api_key):
    """Passes the image to Gemini to get BOTH a short UI label and detailed LLM context."""
    genai.configure(api_key=api_key)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    
    vlm_prompt = (
        "Analyze this poster and return ONLY a raw JSON object with two keys. Do not use markdown blocks:\n"
        '1. "short_aesthetic": A 3-5 word summary of the exact aesthetic and typography (e.g., "Vibrant Anime Maximalist").\n'
        '2. "detailed_context": A 2-3 sentence visual analysis of the art style, typography, and hierarchy. '
        "Do not critique the layout, just identify the stylistic intent."
    )
    
    try:
        response = model.generate_content([vlm_prompt, Image.fromarray(rgb_image)])
        
        # Clean up the output just in case Gemini adds markdown formatting like ```json
        clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        vlm_data = json.loads(clean_text)
        return vlm_data
        
    except Exception as e:
        return {
            "short_aesthetic": "Unknown Aesthetic",
            "detailed_context": f"VLM Error: {str(e)}"
        }


# --- 3. Text LLM Engine (Qwen) ---

def generate_critique(opencv_data, detailed_context, api_key):
    """Sends quantitative data + qualitative context to Qwen 2.5 with expanded prompting."""
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=api_key)
    
    system_prompt = (
        "You are a Senior Art Director providing deep, comprehensive feedback on a poster design. "
        "Do NOT describe what the poster currently looks like (the user already knows). "
        "Skip all introductory and concluding remarks to save space. "
        "Provide 3 to 4 DETAILED, highly professional paragraphs (or comprehensive bullet points) of actionable advice. "
        "For each point, explain the underlying design theory, WHY the change is needed based on the metrics, and EXACTLY HOW to execute it while enhancing the intended aesthetic vibe. "
        "Do not just say 'reduce text' or 'change colors'—explain the structural, psychological, or typographical mechanics behind your advice."
    )
    
    user_data = (
        f"Intended Vibe & Context: {detailed_context}\n"
        f"Text Density: {opencv_data['text_area_percentage']}%\n"
        f"Peak Saturation: {opencv_data['vibrancy']}%\n"
        f"Dominant Colors: {opencv_data['color_paragraph']}\n\n"
        "Give me highly detailed, actionable design improvement advice."
    )
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_data}]
    
    try:
        # Bumped max_tokens to 800 to ensure the LLM has room for the expanded critique
        return client.chat_completion(messages=messages, max_tokens=800, temperature=0.4).choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"


# --- 4. Main Application Flow ---

uploaded_file = st.file_uploader("Upload Poster", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    if not hf_key or not gemini_key:
        st.warning("⚠️ Cannot proceed without both API keys.")
        st.stop()
        
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner("Step 1: OpenCV extracting math metrics..."):
        colors = process_colors(image)
        annotated_img, text_area, vibrancy, temp = analyze_text_and_metrics(image)
        
        color_strings = [f"{c['hex']}" for c in colors]
        color_paragraph = ", ".join(color_strings)
        
    with st.spinner("Step 2: Gemini 3.1 Flash Lite checking visual vibe..."):
        vlm_data = get_vibe_check(image, gemini_key)
        
        if "VLM Error" in vlm_data['detailed_context']:
            st.error(f"Gemini failed to analyze the image: {vlm_data['detailed_context']}")
            st.stop()
    
    with st.spinner("Step 3: Qwen 2.5 generating actionable advice..."):
        advice = generate_critique({
            "text_area_percentage": text_area,
            "vibrancy": vibrancy,
            "color_paragraph": color_paragraph
        }, vlm_data['detailed_context'], hf_key)

    # --- UI Dashboard ---
    st.divider()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(annotated_img, channels="BGR", use_container_width=True)
        
    with col2:
        # 1. Aesthetic Block (Using short_aesthetic from VLM)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Detected Aesthetic</div>
                <div style="font-size: 22px; font-weight: bold; color: #4facfe;">{vlm_data['short_aesthetic']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Color Distribution Bar (GitHub Style)
        bar_html = "".join([f'<div style="width: {c["percentage"]}%; background-color: {c["hex"]};"></div>' for c in colors])
        labels_html = " ".join([f'<span style="color:{c["hex"]}; margin-right: 10px;">● {c["hex"]} ({c["percentage"]}%)</span>' for c in colors])
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Color Distribution</div>
                <div style="width: 100%; height: 18px; display: flex; border-radius: 9px; overflow: hidden; margin-bottom: 10px; border: 1px solid #444;">
                    {bar_html}
                </div>
                <div style="font-size: 13px; color: #CCC;">{labels_html}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 3. Peak Saturation
        sat_label = "Highly Saturated" if vibrancy >= 65 else ("Moderate Saturation" if vibrancy >= 30 else "Low Saturation / Muted")
        sat_color = "#ff4b4b" if vibrancy >= 65 else ("#faca2b" if vibrancy >= 30 else "#888888")
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Peak Saturation ({vibrancy}%)</div>
                <div class="metric-value">{sat_label}</div>
                <div style="width: 100%; background-color: #333; border-radius: 5px; height: 10px;">
                    <div style="width: {vibrancy}%; background-color: {sat_color}; height: 10px; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. Text Density
        text_label = "Heavy / Editorial" if text_area >= 40 else ("Balanced" if text_area >= 15 else "Minimalist / Sparse")
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Text Density ({text_area}%)</div>
                <div class="metric-value">{text_label}</div>
                <div style="width: 100%; background-color: #333; border-radius: 5px; height: 10px;">
                    <div style="width: {text_area}%; background-color: #4facfe; height: 10px; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 5. Temperature Bar
        temp_label = "Warm Toned" if temp > 55 else ("Cool Toned" if temp < 45 else "Neutral Toned")
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Visual Temperature</div>
                <div class="metric-value">{temp_label}</div>
                <div style="width: 100%; height: 12px; border-radius: 6px; background: linear-gradient(to right, #2b5876 0%, #4e4376 30%, #b06ab3 50%, #f77062 80%, #fe5196 100%); position: relative; margin-top: 15px;">
                     <div style="position: absolute; left: calc({temp}% - 7px); top: -10px; color: white; font-size: 16px;">▼</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- Advice Section ---
    st.subheader("Art Director's Suggestions")
    st.info(advice)
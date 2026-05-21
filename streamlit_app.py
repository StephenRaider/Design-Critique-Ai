import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from huggingface_hub import InferenceClient
import json
import math

# --- Configuration & UI Setup ---
st.set_page_config(page_title="AI Design Critic", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin-bottom: 15px; color: white; border: 1px solid #333; height: 100%;
    }
    .metric-title {
        font-size: 11px; color: #AAAAAA; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-value { font-size: 15px; font-weight: bold; margin-bottom: 5px; }
    
    /* Segmented 12-Part Color Wheel */
    .wheel-container { 
        position: relative; width: 120px; height: 120px; margin: 0 auto; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        background: conic-gradient(from -15deg, #FFEA00 0deg 30deg, #FF9900 30deg 60deg, #FF5500 60deg 90deg, #FF0000 90deg 120deg, #CC0066 120deg 150deg, #800080 150deg 180deg, #4B0082 180deg 210deg, #0000FF 210deg 240deg, #008080 240deg 270deg, #00CC00 270deg 300deg, #88CC00 300deg 330deg, #CCDD00 330deg 360deg); 
    }
    .wheel-inner { width: 50px; height: 50px; background-color: #1E1E1E; border-radius: 50%; position: absolute; }
    .color-dot { 
        position: absolute; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white; transform: translate(-50%, -50%); box-shadow: 0 0 4px rgba(0,0,0,0.8); z-index: 2; 
    }
    
    /* Upload UI Styling */
    .upload-header { text-align: center; margin-bottom: 10px; }
    .upload-header h3 { margin: 0; color: #4facfe; font-size: 24px;}
    .upload-header p { color: #888; font-size: 14px;}
    </style>
""", unsafe_allow_html=True)

st.title("🎨 AI Poster Design Critic (Flaw Detection Mode)")

try:
    hf_key = st.secrets["HF_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    hf_key, gemini_key = None, None
    st.error("⚠️ Missing API Keys in .streamlit/secrets.toml")

# --- 1. Computer Vision Functions (OpenCV) ---

def check_4axis_balance(mask):
    total_weight = np.sum(mask)
    if total_weight == 0: return True 
    img_h, img_w = mask.shape
    
    left_half = np.sum(mask[:, :img_w//2]) / total_weight
    top_half = np.sum(mask[:img_h//2, :]) / total_weight
    
    return (0.45 <= left_half <= 0.55) or (0.45 <= top_half <= 0.55)

def process_colors(image, k=5):
    blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    image_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(image_rgb, (150, 150), interpolation=cv2.INTER_AREA).reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    counts = Counter(kmeans.labels_)
    total_pixels = len(pixels)
    
    colors_info = []
    for i in counts.keys():
        r, g, b = [int(v) for v in kmeans.cluster_centers_[i]]
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        
        c_dict = {
            "rgb": [r, g, b], "hex": '#{:02x}{:02x}{:02x}'.format(r, g, b).upper(),
            "hue": int(hsv_color[0] * 2), "sat": hsv_color[1], "val": hsv_color[2],
            "percentage": round((counts[i] / total_pixels) * 100, 1)
        }
        colors_info.append(c_dict)
        
    colors_info.sort(key=lambda x: x['percentage'], reverse=True)
    
    bg_color = colors_info[0] if colors_info[0]['percentage'] > 60 else None
    palette = colors_info[1:] if bg_color else colors_info
    
    # Calculate relative percentages for the foreground palette
    rem_total = sum(c['percentage'] for c in palette)
    for c in palette:
        c['rel_percentage'] = round((c['percentage'] / (rem_total + 1e-5)) * 100, 1)
    
    core_colors = [c for c in palette if c['sat'] > 40 and 30 < c['val'] < 240]
    harmony = "Desaturated / Muted"
    
    if len(core_colors) == 0:
        harmony = "Monotonous / Desaturated"
    elif len(core_colors) == 1:
        harmony = "Monochromatic"
    else:
        hues = sorted([c['hue'] for c in core_colors])
        max_diff = max([abs(hues[i] - hues[i-1]) for i in range(1, len(hues))] + [(360 - hues[-1] + hues[0])])
        if max_diff < 50:
            harmony = "Analogous"
        elif max_diff > 140:
            harmony = "Complementary"
        else:
            harmony = "Split Complementary / Triadic"
            
    return bg_color, palette, harmony

def analyze_balance_and_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    
    img_h, img_w = image.shape[:2]
    
    # Balance
    struct_balanced = check_4axis_balance(thresh > 0)
    color_balanced = check_4axis_balance(s > 50)
    imbalances = []
    if not struct_balanced: imbalances.append("Structural")
    if not color_balanced: imbalances.append("Color Weight")
    balance_str = " & ".join(imbalances) + " Imbalance Detected" if imbalances else "Perfectly Balanced"

    # Text Area
    text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    contours, _ = cv2.findContours(cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 10 and h < img_h * 0.3 and w < img_w * 0.9 and (w / float(h)) > 1.2: 
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
    text_area_percentage = round((cv2.countNonZero(text_mask) / (img_h * img_w)) * 100, 1)
    
    # Vibrancy & Tone (Temperature)
    vibrancy_score = round(np.percentile(s, 90) / 255.0 * 100, 1)
    mean_b = image[:, :, 0].mean()
    mean_r = image[:, :, 2].mean()
    temp_score = round((mean_r / (mean_r + mean_b + 1e-5)) * 100, 1)
    
    return annotated_image, balance_str, text_area_percentage, vibrancy_score, temp_score

# --- 2. Vision-Language Model (Gemini) ---

def get_vlm_insights(opencv_image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    vlm_prompt = (
        "Analyze this poster and return ONLY a raw JSON object with these exact keys. Do not critique, just observe:\n"
        '"aesthetic": 3-5 word summary.\n'
        '"detected_font": Describe the font style briefly.\n'
        '"font_suitability": Does the font fit the aesthetic? Answer strictly "Good choice" OR "Recommend: [Better Font Type]".\n'
        '"unity": (1 sentence) Cohesion assessment.\n'
        '"focal_point": (1 sentence) What draws the eye first.\n'
        '"variety": (1 sentence) Assessment of visual variety.\n'
        '"recommended_palette": An array of exactly 5 hex color codes (e.g., ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"]) forming a beautiful, professional palette (like from Coolors) that elevates the intended aesthetic and fixes any current color clashing.'
    )
    try:
        response = model.generate_content([vlm_prompt, Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))])
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except:
        return {
            "aesthetic": "Error", "font_suitability": "Error", "unity": "", "focal_point": "", "variety": "", 
            "recommended_palette": ["#1E1E1E", "#333333", "#555555", "#777777", "#999999"]
        }

# --- 3. Text LLM Engine (Qwen) ---

def generate_critique(metrics, vlm_data, api_key):
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=api_key)
    system_prompt = (
        "You are an Art Director critiquing a poster. "
        "CRITICAL INSTRUCTION: You MUST start your response with the exact text 'MAIN_REC: ' followed by a single, SHORT, ACTIONABLE sentence stating the top recommendation to improve the design (e.g., 'Balance visual structure by evenly distributing elements.'). Do not describe the flaw here, just tell them what to do. "
        "After that sentence, leave a double line break, then output 3 highly detailed, actionable paragraphs explaining WHY the design needs improvement mathematically/visually and EXACTLY HOW to fix it."
    )
    user_data = (
        f"VLM Context: Aesthetic: {vlm_data.get('aesthetic')}. Unity: {vlm_data.get('unity')}. Variety: {vlm_data.get('variety')}.\n"
        f"OpenCV Metrics: Balance Status: {metrics['balance']}. Text Density: {metrics['text_area']}%. Peak Saturation: {metrics['vibrancy']}%. Color Harmony: {metrics['harmony']}."
    )
    try:
        return client.chat_completion(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_data}], max_tokens=800, temperature=0.4).choices[0].message.content
    except Exception as e:
        return f"MAIN_REC: API Error occurred.\n\n{str(e)}"

# --- 4. Main Application Flow ---

# Stylized Upload UI - Single clean uploader
st.markdown('<div class="upload-header"><h3>Upload Image</h3><p>Drag & drop a file, paste an image (Ctrl+V), or click to browse</p></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None and hf_key and gemini_key:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Analyzing mass, core colors, and distribution..."):
        bg_color, palette, harmony = process_colors(image)
        annotated_img, balance_str, text_area, vibrancy, temp = analyze_balance_and_metrics(image)
        
    with st.spinner("VLM assessing typography, palette, and unity..."):
        vlm_data = get_vlm_insights(image, gemini_key)
    
    with st.spinner("Qwen drafting final critique..."):
        raw_advice = generate_critique({"balance": balance_str, "text_area": text_area, "vibrancy": vibrancy, "harmony": harmony}, vlm_data, hf_key)
        
        main_rec = "Improve overall composition."
        detailed_advice = raw_advice
        if "MAIN_REC:" in raw_advice:
            parts = raw_advice.split("\n\n", 1)
            main_rec = parts[0].replace("MAIN_REC:", "").replace("**", "").strip()
            detailed_advice = parts[1] if len(parts) > 1 else "See main recommendation."

    # --- UI Dashboard ---
    st.divider()
    col_img, col_data, col_color = st.columns([1.2, 1, 1])
    
    with col_img:
        st.image(annotated_img, channels="BGR", use_container_width=True)
        
    with col_data:
        # The Main Recommendation Block
        st.markdown(f'<div class="metric-card" style="border: 1px solid #4facfe; background-color: #0b1a26;"><div class="metric-title" style="color: #4facfe;">💡 Top Recommendation</div><div style="font-size: 16px; font-weight: bold; color: white;">{main_rec}</div></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card"><div class="metric-title">Detected Aesthetic</div><div class="metric-value" style="color: #4facfe;">{vlm_data.get("aesthetic", "Unknown")}</div></div>', unsafe_allow_html=True)
        
        bal_color = "#4facfe" if "Perfectly" in balance_str else "#ff4b4b"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Composition Status</div><div class="metric-value" style="color: {bal_color};">{balance_str}</div></div>', unsafe_allow_html=True)
        
        font_str = vlm_data.get("font_suitability", "Unknown")
        font_color = "#4facfe" if "Good" in font_str else "#faca2b"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Typography Assessment</div><div class="metric-value" style="color: {font_color};">{font_str}</div></div>', unsafe_allow_html=True)

    with col_color:
        # Tone/Temperature Bar
        temp_label = "Warm Toned" if temp > 55 else ("Cool Toned" if temp < 45 else "Neutral Toned")
        st.markdown(f'<div class="metric-card"><div class="metric-title">Visual Temperature</div><div class="metric-value">{temp_label}</div><div style="width: 100%; height: 12px; border-radius: 6px; background: linear-gradient(to right, #2b5876 0%, #4e4376 30%, #b06ab3 50%, #f77062 80%, #fe5196 100%); position: relative; margin-top: 15px;"><div style="position: absolute; left: calc({temp}% - 7px); top: -10px; color: white; font-size: 16px;">▼</div></div></div>', unsafe_allow_html=True)

        if bg_color:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Dominant Background ({bg_color["percentage"]}%)</div><div style="width: 100%; height: 20px; background-color: {bg_color["hex"]}; border-radius: 5px; border: 1px solid #555;"></div></div>', unsafe_allow_html=True)
        
        # Relative Color Distribution Bar using rel_percentage!
        bar_html = "".join([f'<div style="width: {c.get("rel_percentage", c["percentage"])}%; background-color: {c["hex"]};"></div>' for c in palette])
        labels_html = "<br>".join([f'<span style="color:{c["hex"]};">● {c["hex"]} ({c.get("rel_percentage", c["percentage"])}%)</span>' for c in palette])
        st.markdown(f'<div class="metric-card"><div class="metric-title">Foreground Palette ({harmony})</div><div style="width: 100%; height: 18px; display: flex; border-radius: 9px; overflow: hidden; margin-bottom: 10px; border: 1px solid #444;">{bar_html}</div><div style="font-size: 12px; color: #CCC;">{labels_html}</div></div>', unsafe_allow_html=True)
        
        # Pull the 5-Color Recommended Palette securely from Gemini's JSON
        rec_hexes = vlm_data.get("recommended_palette", ["#333333", "#555555", "#777777", "#999999", "#BBBBBB"])
        if not isinstance(rec_hexes, list) or len(rec_hexes) != 5:
            rec_hexes = ["#333333", "#555555", "#777777", "#999999", "#BBBBBB"]
            
        rec_html = "".join([f'<div style="flex: 1; height: 25px; background-color: {hx}; margin-right: 5px; border-radius: 4px; border: 1px solid #555;"></div>' for hx in rec_hexes])
        st.markdown(f'<div class="metric-card"><div class="metric-title">Suggested True Palette</div><div style="display: flex; width: 100%;">{rec_html}</div></div>', unsafe_allow_html=True)
        
        wheel_dots = ""
        for c in palette:
            rad = math.radians(c['hue'] - 90)
            x, y = 50 + 35 * math.cos(rad), 50 + 35 * math.sin(rad)
            wheel_dots += f'<div class="color-dot" style="left: {x}%; top: {y}%; background-color: {c["hex"]};"></div>'
        st.markdown(f'<div class="metric-card"><div class="metric-title">Color Wheel Placement</div><div class="wheel-container"><div class="wheel-inner"></div>{wheel_dots}</div></div>', unsafe_allow_html=True)

    # --- Advice Section ---
    st.subheader("Art Director Breakdown")
    st.info(detailed_advice)
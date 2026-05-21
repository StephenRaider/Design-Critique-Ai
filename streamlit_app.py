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
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        border: 1px solid #333;
        height: 100%;
    }
    .metric-title {
        font-size: 12px;
        color: #AAAAAA;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    
    /* NEW: Segmented 12-Part Color Wheel */
    .wheel-container { 
        position: relative; 
        width: 120px; 
        height: 120px; 
        margin: 0 auto; 
        border-radius: 50%; 
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(
            from -15deg,
            #FFEA00 0deg 30deg,     /* Yellow */
            #FF9900 30deg 60deg,    /* Yellow-Orange */
            #FF5500 60deg 90deg,    /* Orange */
            #FF0000 90deg 120deg,   /* Red */
            #CC0066 120deg 150deg,  /* Red-Violet */
            #800080 150deg 180deg,  /* Violet */
            #4B0082 180deg 210deg,  /* Blue-Violet */
            #0000FF 210deg 240deg,  /* Blue */
            #008080 240deg 270deg,  /* Blue-Green */
            #00CC00 270deg 300deg,  /* Green */
            #88CC00 300deg 330deg,  /* Yellow-Green */
            #CCDD00 330deg 360deg   /* Lime-Yellow */
        ); 
    }
    /* NEW: The hollow center cutout */
    .wheel-inner {
        width: 50px;
        height: 50px;
        background-color: #1E1E1E; /* Matches the metric card background */
        border-radius: 50%;
        position: absolute;
    }
    .color-dot { 
        position: absolute; 
        width: 16px; 
        height: 16px; 
        border-radius: 50%; 
        border: 2px solid white; 
        transform: translate(-50%, -50%); 
        box-shadow: 0 0 4px rgba(0,0,0,0.8);
        z-index: 2; /* Keeps dots above the inner circle */
    }
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
    """Helper function to test if a binary mask is balanced across 4 axes."""
    total_weight = np.sum(mask)
    if total_weight == 0:
        return True # Avoid division by zero if blank
        
    img_h, img_w = mask.shape
    
    left_half = mask[:, :img_w//2]
    top_half = mask[:img_h//2, :]
    diag1_mask = np.triu(np.ones((img_h, img_w), dtype=bool))
    diag2_mask = np.fliplr(diag1_mask)
    
    ratios = [
        np.sum(left_half) / total_weight,
        np.sum(top_half) / total_weight,
        np.sum(mask & diag1_mask) / total_weight,
        np.sum(mask & diag2_mask) / total_weight
    ]
    
    # If any ratio is between 40% and 60%, it's balanced on at least one axis
    return any(0.40 <= r <= 0.60 for r in ratios)

def process_colors(image, k=5):
    """Extracts colors, isolates BG if > 60%, and determines harmony."""
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
        
        # Calculate Hue for harmony/wheel (0-360)
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = int(hsv_color[0] * 2)
        
        percentage = round((counts[i] / total_pixels) * 100, 1)
        colors_info.append({"rgb": [r, g, b], "hex": hex_code, "percentage": percentage, "hue": hue})
        
    colors_info.sort(key=lambda x: x['percentage'], reverse=True)
    
    bg_color = None
    palette = colors_info
    
    # 60% Background Rule
    if colors_info[0]['percentage'] > 60:
        bg_color = colors_info[0]
        palette = colors_info[1:]
        rem_total = sum(c['percentage'] for c in palette)
        for c in palette:
            c['rel_percentage'] = round((c['percentage'] / (rem_total + 1e-5)) * 100, 1)
    else:
        for c in palette:
            c['rel_percentage'] = c['percentage']

    # Estimate Color Harmony based on Hues
    hues = sorted([c['hue'] for c in palette])
    harmony = "Custom / Uncategorized"
    if len(hues) >= 2:
        max_diff = max(abs(hues[i] - hues[i-1]) for i in range(1, len(hues)))
        max_diff = max(max_diff, 360 - hues[-1] + hues[0]) # Wrap around
        
        if max_diff < 40: harmony = "Analogous / Monochromatic"
        elif 160 <= max_diff <= 200: harmony = "Complementary"
        elif 120 <= max_diff < 160: harmony = "Split Complementary / Triadic"
        elif len(hues) >= 4: harmony = "Tetradic / Complex"
        
    return bg_color, palette, harmony

def analyze_balance_and_metrics(image):
    """Calculates text density, saturation, and the Dual 4-Axis Balance Tests."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    
    img_h, img_w = image.shape[:2]
    
    # 1. Structural Balance Test (Uses grayscale mass)
    struct_mask = thresh > 0 
    struct_is_balanced = check_4axis_balance(struct_mask)
    struct_balance = "Structurally Balanced" if struct_is_balanced else "Severely Unbalanced (Mass)"
    
    # 2. Color Balance Test (Uses highly saturated pixels)
    color_mask = s > 40 # Threshold for vibrant colors, ignores white/black/gray
    color_is_balanced = check_4axis_balance(color_mask)
    color_balance = "Color is Well-Distributed" if color_is_balanced else "Color Weight is Lopsided"

    # 3. Text Area
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_image = image.copy()
    text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 15 and h > 10) and (h < img_h * 0.3) and (w < img_w * 0.9):
            if (w / float(h)) > 1.2: 
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
                
    text_area_percentage = round((cv2.countNonZero(text_mask) / (img_h * img_w)) * 100, 1)
            
    # 4. Vibrancy
    vibrancy_score = round(np.percentile(s, 90) / 255.0 * 100, 1)
    
    return annotated_image, struct_balance, color_balance, text_area_percentage, vibrancy_score


# --- 2. Vision-Language Model (Gemini) ---

def get_vlm_insights(opencv_image, api_key):
    genai.configure(api_key=api_key)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    
    vlm_prompt = (
        "Analyze this poster and return ONLY a raw JSON object with these exact keys:\n"
        '"aesthetic": 3-5 word summary (e.g., "Minimalist Swiss Corporate").\n'
        '"unity": Are the elements cohesive, or do they look like unrelated items slapped together? (1 sentence).\n'
        '"focal_point": What exact element draws the eye first? Is it legible? (1 sentence).\n'
        '"variety": Is there good variation in font sizes/weights/colors, or is it visually monotonous and boring? (1 sentence).'
    )
    
    try:
        response = model.generate_content([vlm_prompt, Image.fromarray(rgb_image)])
        clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"aesthetic": "Error", "unity": "Error", "focal_point": "Error", "variety": "Error"}


# --- 3. Text LLM Engine (Qwen) ---

def generate_critique(metrics, vlm_data, api_key):
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=api_key)
    
    system_prompt = (
        "You are a strict, expert Art Director critiquing a flawed poster. "
        "Skip all intro/outro text. Output ONLY 3-4 highly detailed, actionable paragraphs. "
        "For each point, identify the specific flaw from the provided data (e.g., unbalanced layout, lopsided color weight, lack of variety, wrong focal point). "
        "Explain WHY it fails graphic design theory, and tell the designer EXACTLY HOW to fix it to match the intended aesthetic."
    )
    
    user_data = (
        f"VLM Visual Insights:\n"
        f"- Intended Aesthetic: {vlm_data.get('aesthetic')}\n"
        f"- Unity: {vlm_data.get('unity')}\n"
        f"- Focal Point/Contrast: {vlm_data.get('focal_point')}\n"
        f"- Visual Variety: {vlm_data.get('variety')}\n\n"
        f"OpenCV Mathematical Metrics:\n"
        f"- Structural Balance: {metrics['struct_balance']}\n"
        f"- Color Balance: {metrics['color_balance']}\n"
        f"- Text Density: {metrics['text_area']}%\n"
        f"- Peak Saturation: {metrics['vibrancy']}%\n"
        f"- Color Palette Harmony: {metrics['harmony']}\n"
        "Critique the flaws strictly based on these explicit readouts."
    )
    
    try:
        return client.chat_completion(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_data}], max_tokens=800, temperature=0.4).choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"


# --- 4. Main Application Flow ---

uploaded_file = st.file_uploader("Upload Poster", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None and hf_key and gemini_key:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner("Analyzing mass, color weight, and distribution..."):
        bg_color, palette, harmony = process_colors(image)
        annotated_img, struct_balance, color_balance, text_area, vibrancy = analyze_balance_and_metrics(image)
        
    with st.spinner("VLM determining unity and focal points..."):
        vlm_data = get_vlm_insights(image, gemini_key)
    
    with st.spinner("Qwen generating strict critique..."):
        advice = generate_critique({
            "struct_balance": struct_balance, "color_balance": color_balance, 
            "text_area": text_area, "vibrancy": vibrancy, "harmony": harmony
        }, vlm_data, hf_key)

    # --- UI Dashboard ---
    st.divider()
    
    col_img, col_data, col_color = st.columns([1.2, 1, 1])
    
    with col_img:
        st.image(annotated_img, channels="BGR", use_container_width=True)
        
    with col_data:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Detected Aesthetic</div><div class="metric-value" style="color: #4facfe;">{vlm_data.get("aesthetic", "Unknown")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-title">Structural Balance</div><div class="metric-value">{struct_balance}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-title">Color Balance</div><div class="metric-value">{color_balance}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-title">VLM Insight: Unity</div><div style="font-size: 14px;">{vlm_data.get("unity", "N/A")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-title">VLM Insight: Focal Point</div><div style="font-size: 14px;">{vlm_data.get("focal_point", "N/A")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-title">VLM Insight: Variety</div><div style="font-size: 14px;">{vlm_data.get("variety", "N/A")}</div></div>', unsafe_allow_html=True)

    with col_color:
        # Background Color Rule Check
        if bg_color:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Dominant Background ({bg_color["percentage"]}%)</div><div style="width: 100%; height: 20px; background-color: {bg_color["hex"]}; border-radius: 5px; border: 1px solid #555;"></div></div>', unsafe_allow_html=True)
        
        # Relative Color Distribution Bar
        bar_html = "".join([f'<div style="width: {c["rel_percentage"]}%; background-color: {c["hex"]};"></div>' for c in palette])
        labels_html = "<br>".join([f'<span style="color:{c["hex"]};">● {c["hex"]} ({c["rel_percentage"]}%)</span>' for c in palette])
        st.markdown(f'<div class="metric-card"><div class="metric-title">Foreground Palette ({harmony})</div><div style="width: 100%; height: 18px; display: flex; border-radius: 9px; overflow: hidden; margin-bottom: 10px; border: 1px solid #444;">{bar_html}</div><div style="font-size: 12px; color: #CCC;">{labels_html}</div></div>', unsafe_allow_html=True)
        
        # RYB Color Wheel UI
        wheel_dots = ""
        for c in palette:
            rad = math.radians(c['hue'] - 90) # Offset so 0 is top
            # Changed radius from 40 to 35 to center the dots in the color band
            x, y = 50 + 35 * math.cos(rad), 50 + 35 * math.sin(rad)
            wheel_dots += f'<div class="color-dot" style="left: {x}%; top: {y}%; background-color: {c["hex"]};"></div>'
            
        # Added wheel-inner div to create the donut shape
        st.markdown(f'<div class="metric-card"><div class="metric-title">Color Wheel Placement</div><div class="wheel-container"><div class="wheel-inner"></div>{wheel_dots}</div></div>', unsafe_allow_html=True)

    # --- Advice Section ---
    st.subheader("Strict Art Director Critique")
    st.info(advice)
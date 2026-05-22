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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Design Critic", layout="wide")

st.markdown("""
<style>
/* ── Global dark canvas ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0D0D0D !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #111 !important; }

/* ── Metric cards (results dashboard) ──────────────────────── */
.metric-card {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    color: white;
    border: 1px solid #2a2a2a;
}
.metric-title {
    font-size: 11px; color: #888; margin-bottom: 8px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px;
}
.metric-value { font-size: 15px; font-weight: bold; margin-bottom: 5px; }

/* ── Landing page ────────────────────────────────────────────── */
.landing-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 88vh;
    padding: 0 4vw;
    gap: 40px;
}
.landing-left {
    flex: 1 1 45%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 20px;
}
.landing-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4facfe;
    margin: 0;
}
.landing-title {
    font-size: clamp(38px, 5vw, 72px);
    font-weight: 900;
    line-height: 1.0;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -1.5px;
}
.landing-title span { color: #4facfe; }
.landing-sub {
    font-size: 15px;
    color: #666;
    line-height: 1.7;
    margin: 0;
    max-width: 420px;
}
.landing-pills {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px;
}
.pill {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    padding: 5px 12px; border-radius: 20px;
    border: 1px solid #2a2a2a; color: #888;
    text-transform: uppercase;
}
.landing-right {
    flex: 0 0 420px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.upload-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 36px 32px 28px;
}
.upload-card-title {
    font-size: 13px; font-weight: 700; color: #fff;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;
}
.upload-card-sub {
    font-size: 12px; color: #555; margin-bottom: 20px;
}
.stat-row {
    display: flex; gap: 10px; margin-top: 4px;
}
.stat-box {
    flex: 1; background: #111; border: 1px solid #222;
    border-radius: 12px; padding: 14px 12px; text-align: center;
}
.stat-box .num { font-size: 20px; font-weight: 800; color: #4facfe; }
.stat-box .lbl { font-size: 10px; color: #555; text-transform: uppercase;
                  letter-spacing: 1px; margin-top: 2px; }

/* ── Streamlit uploader blending ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: #0f0f0f !important;
    border: 1.5px dashed #2e2e2e !important;
    border-radius: 12px !important;
    padding: 24px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4facfe !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #555 !important; font-size: 13px !important;
}

/* ── Color wheel ────────────────────────────────────────────── */
.wheel-container {
    position: relative; width: 120px; height: 120px;
    margin: 0 auto; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(from -15deg,
        #FFEA00 0deg 30deg, #FF9900 30deg 60deg, #FF5500 60deg 90deg,
        #FF0000 90deg 120deg, #CC0066 120deg 150deg, #800080 150deg 180deg,
        #4B0082 180deg 210deg, #0000FF 210deg 240deg, #008080 240deg 270deg,
        #00CC00 270deg 300deg, #88CC00 300deg 330deg, #CCDD00 330deg 360deg);
}
.wheel-inner {
    width: 50px; height: 50px; background-color: #1E1E1E;
    border-radius: 50%; position: absolute;
}
.color-dot {
    position: absolute; width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid white; transform: translate(-50%, -50%);
    box-shadow: 0 0 4px rgba(0,0,0,0.8); z-index: 2;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────────────────────────────────────────
try:
    hf_key     = st.secrets["HF_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    hf_key = gemini_key = None
    st.error("⚠️ Missing API Keys in .streamlit/secrets.toml")


# ─────────────────────────────────────────────────────────────────────────────
# 1. OPENCV — BALANCE, METRICS, COLOR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def check_4axis_balance(mask):
    total_weight = float(np.sum(mask))
    if total_weight == 0:
        return True, []

    img_h, img_w = mask.shape
    rows, cols   = np.ogrid[:img_h, :img_w]
    LO, HI       = 0.40, 0.60
    imbalances   = []

    left_ratio = np.sum(mask[:, : img_w // 2]) / total_weight
    if not (LO <= left_ratio <= HI):
        imbalances.append("Left-Heavy" if left_ratio > HI else "Right-Heavy")

    top_ratio = np.sum(mask[: img_h // 2, :]) / total_weight
    if not (LO <= top_ratio <= HI):
        imbalances.append("Top-Heavy" if top_ratio > HI else "Bottom-Heavy")

    upper_right_mask = (cols * img_h) > (rows * img_w)
    ur_ratio = np.sum(mask[upper_right_mask]) / total_weight
    if not (LO <= ur_ratio <= HI):
        imbalances.append("Top-Right Heavy" if ur_ratio > HI else "Bottom-Left Heavy")

    upper_left_mask = (cols * img_h + rows * img_w) < (img_w * img_h)
    ul_ratio = np.sum(mask[upper_left_mask]) / total_weight
    if not (LO <= ul_ratio <= HI):
        imbalances.append("Top-Left Heavy" if ul_ratio > HI else "Bottom-Right Heavy")

    return len(imbalances) == 0, imbalances


def process_colors(image, k=5):
    blurred   = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    image_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    pixels    = cv2.resize(image_rgb, (150, 150), interpolation=cv2.INTER_AREA).reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    counts = Counter(kmeans.labels_)
    total  = len(pixels)

    colors_info = []
    for i in counts.keys():
        r, g, b   = [int(v) for v in kmeans.cluster_centers_[i]]
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        colors_info.append({
            "rgb": [r, g, b],
            "hex": "#{:02x}{:02x}{:02x}".format(r, g, b).upper(),
            "hue": int(hsv_color[0] * 2),
            "sat": hsv_color[1],
            "val": hsv_color[2],
            "percentage": round((counts[i] / total) * 100, 1),
        })

    colors_info.sort(key=lambda x: x["percentage"], reverse=True)

    bg_color = colors_info[0] if colors_info[0]["percentage"] > 60 else None
    palette  = colors_info[1:] if bg_color else colors_info

    rem_total = sum(c["percentage"] for c in palette)
    for c in palette:
        c["rel_percentage"] = round((c["percentage"] / (rem_total + 1e-5)) * 100, 1)

    core_colors = [c for c in palette if c["sat"] > 40 and 20 < c["val"] < 240]
    if len(core_colors) < 2:
        core_colors = [c for c in palette if c["sat"] > 15 and 15 < c["val"] < 245]

    harmony = "Desaturated / Neutral"
    if len(core_colors) <= 1:
        harmony = "Monochromatic / Desaturated"
    else:
        hues    = sorted([c["hue"] for c in core_colors])
        max_gap = max(
            [abs(hues[i] - hues[i - 1]) for i in range(1, len(hues))]
            + [(360 - hues[-1] + hues[0])]
        )
        span = 360 - max_gap
        if span < 60:
            harmony = "Analogous"
        elif 140 <= span <= 220:
            harmony = "Complementary"
        else:
            harmony = "Triadic / Complex"

    return bg_color, palette, harmony, core_colors


def analyze_balance_and_metrics(image):
    # Standard grayscale for balance checking
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    hsv       = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _   = cv2.split(hsv)
    img_h, img_w = image.shape[:2]

    struct_ok, struct_issues = check_4axis_balance(thresh > 0)
    color_ok,  color_issues  = check_4axis_balance(s > 100)

    problem_parts = []
    if not struct_ok:
        problem_parts.append("Structure: " + ", ".join(struct_issues))
    if not color_ok:
        problem_parts.append("Color Weight: " + ", ".join(color_issues))

    balance_str = "\n".join(problem_parts) if problem_parts else "Balanced"

    # LAB Color Space for Robust Text Extraction
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    _, text_thresh = cv2.threshold(l_channel, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
    dilated   = cv2.dilate(text_thresh, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 15 and h > 10 and h < img_h * 0.3
                and w < img_w * 0.9 and (w / float(h)) > 1.2):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)

    cv_text_area   = round((cv2.countNonZero(text_mask) / (img_h * img_w)) * 100, 1)
    vibrancy_score = round(np.percentile(s, 90) / 255.0 * 100, 1)
    mean_b = image[:, :, 0].mean()
    mean_r = image[:, :, 2].mean()
    temp_score = round((mean_r / (mean_r + mean_b + 1e-5)) * 100, 1)

    return annotated, balance_str, cv_text_area, vibrancy_score, temp_score


# ─────────────────────────────────────────────────────────────────────────────
# 2. GEMINI VLM — HIERARCHY, FOCAL POINT, TYPOGRAPHY, PALETTE JUDGMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_vlm_insights(opencv_image, api_key, palette, harmony_type):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.1-flash-lite")

    # Build a hex list string so Gemini sees the actual colours
    hex_list = ", ".join([c["hex"] for c in palette[:5]])

    vlm_prompt = (
        f"You are a senior art director. Analyze this image and return ONLY a raw JSON object "
        f"with these exact keys — no markdown, no backticks, no explanation:\n\n"
        f'"is_valid_poster": true if the image is a graphic design, poster, flyer, layout, or typographic piece. false if it is a random photograph, selfie, or completely unrelated non-design image.\n'
        f'"aesthetic": 3-5 word summary of the overall visual style.\n'
        f'"detected_font": Brief description of the font style used (e.g. Bold Condensed Sans-Serif).\n'
        f'"font_suitability": You are a demanding art director. Only say "Good choice" if the font '
        f'is genuinely exceptional and excellently matched to the aesthetic. Otherwise identify a specific '
        f'weakness (wrong mood, legibility issue, style clash) and respond with '
        f'"Recommend: [specific better font type + reason in max 8 words]". '
        f'Default toward recommending an improvement.\n'
        f'"has_focal_point": true if there is a clear element that grabs attention first, false otherwise.\n'
        f'"focal_point": 1 sentence. What draws the eye first. If none, say "No clear anchor element".\n'
        f'"hierarchy_score": Integer 1-10 (1=flat/monotonous, 10=dynamic size/weight contrast).\n'
        f'"estimated_text_density": Integer from 0 to 100 estimating the percentage of the canvas covered by text elements.\n'
        f'"top_visual_mistake": 1 short actionable sentence addressing the absolute most glaring visual, layout, or stylistic flaw you see (e.g. "Reduce the harsh drop shadow on the main title", "Align the scattered text blocks to a single grid"). Be highly specific to the image flaws.\n'
        f'"is_amateur_or_ai": true if the design looks amateurish, inexperienced, or has obvious AI-generated text/layout artifacts, false if it looks reasonably professional.\n'
        f'"amateur_critique": If is_amateur_or_ai is true, provide 1 hard, direct sentence explaining exactly why it looks amateur or AI-generated. If false, output null.\n'
        f'"palette_verdict": Critically judge the current palette [{hex_list}] which follows a '
        f'{harmony_type} scheme. Is it working well for this aesthetic and design? '
        f'Reply with exactly one of: "Good Palette" OR a single sentence critique starting with '
        f'"Needs improvement:" explaining what is wrong (e.g. clashing, muddy, low contrast, wrong mood).\n'
        f'"suggested_palette": ONLY populate this if palette_verdict starts with "Needs improvement". '
        f'Return an object with: "type" (one of: Analogous / Complementary / Triadic / '
        f'Split-Complementary / Monochromatic / Tetradic), '
        f'"colors" (array of exactly 5 hex codes forming a beautiful palette of that type '
        f'suited to this design\'s aesthetic), '
        f'"rationale" (max 10 words explaining the colour theory choice). '
        f'If palette_verdict is "Good Palette", set suggested_palette to null.'
    )

    try:
        response = model.generate_content([
            vlm_prompt,
            Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        ])
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "is_valid_poster": True,
            "aesthetic": "Analysis Error",
            "detected_font": "Unknown",
            "font_suitability": "Error",
            "has_focal_point": False,
            "focal_point": f"VLM call failed: {e}",
            "hierarchy_score": 5,
            "estimated_text_density": 15,
            "top_visual_mistake": "Improve overall composition.",
            "is_amateur_or_ai": False,
            "amateur_critique": None,
            "palette_verdict": "Good Palette",
            "suggested_palette": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. QWEN LLM — ART DIRECTOR CRITIQUE
# ─────────────────────────────────────────────────────────────────────────────

def generate_critique(metrics, vlm_data, api_key):
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=api_key)
    system_prompt = (
        "You are an Art Director critiquing a poster. "
        "CRITICAL INSTRUCTION: You MUST start your response with the exact text 'MAIN_REC: ' "
        "followed by a single, SHORT, ACTIONABLE sentence stating the top recommendation to improve the design. "
        "Base this MAIN_REC directly on the 'Top Visual Mistake' provided in the context to make it highly specific to this poster. "
        "After that sentence, leave a double line break, then output 3 highly detailed, actionable paragraphs "
        "explaining WHY the design needs improvement mathematically/visually and EXACTLY HOW to fix it."
    )
    user_data = (
        f"VLM Context: Aesthetic: {vlm_data.get('aesthetic')}. "
        f"Top Visual Mistake: {vlm_data.get('top_visual_mistake')}. "
        f"Has Focal Point: {vlm_data.get('has_focal_point')}. "
        f"Hierarchy Score: {vlm_data.get('hierarchy_score')}/10. "
        f"Palette Verdict: {vlm_data.get('palette_verdict')}.\n"
        f"OpenCV Metrics: Balance Status: {metrics['balance']}. "
        f"Text Density: {metrics['text_area']}%. "
        f"Peak Saturation: {metrics['vibrancy']}%. "
        f"Color Harmony: {metrics['harmony']}."
    )
    try:
        return client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_data},
            ],
            max_tokens=800, temperature=0.4,
        ).choices[0].message.content
    except Exception as e:
        return f"MAIN_REC: API Error occurred.\n\n{str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. LANDING UI (shown before image upload)
# ─────────────────────────────────────────────────────────────────────────────

def render_landing(file_uploader_key="uploader"):
    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        st.markdown("""
        <div class="landing-left" style="padding-top: 18vh;">
            <p class="landing-eyebrow">Powered by CV + Gemini + Qwen</p>
            <h1 class="landing-title">
                AI Design<br><span>Critique</span><br>Companion
            </h1>
            <p class="landing-sub">
                Upload any poster and get instant, data-driven design feedback —
                balance analysis, colour theory, typography assessment, and
                a full art-director critique in seconds.
            </p>
            <div class="landing-pills">
                <span class="pill">4-Axis Balance</span>
                <span class="pill">WCAG Contrast</span>
                <span class="pill">Color Theory</span>
                <span class="pill">Typography</span>
                <span class="pill">Focal Point</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div style="padding-top: 22vh;">
            <div class="upload-card">
                <p class="upload-card-title">Drop your poster</p>
                <p class="upload-card-sub">JPG, PNG or JPEG · Max 200 MB</p>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="Upload poster",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed",
            key=file_uploader_key,
        )

        st.markdown("""
            </div>
            <div class="stat-row">
                <div class="stat-box"><div class="num">4</div><div class="lbl">Balance Axes</div></div>
                <div class="stat-box"><div class="num">7</div><div class="lbl">Design Metrics</div></div>
                <div class="stat-box"><div class="num">3</div><div class="lbl">AI Layers</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return uploaded


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN APP FLOW
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = render_landing()

if uploaded_file is not None and hf_key and gemini_key:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Analysing mass, colours, and distribution…"):
        bg_color, palette, harmony, core_colors = process_colors(image)
        annotated_img, balance_str, cv_text_area, vibrancy, temp = analyze_balance_and_metrics(image)
        dominant_hex = palette[0]["hex"] if palette else "#FFFFFF"

    with st.spinner("VLM assessing hierarchy, focal points, typography, and validation…"):
        vlm_data = get_vlm_insights(image, gemini_key, palette, harmony)

    # Validation Safety Gate
    if not vlm_data.get("is_valid_poster", True):
        st.error("🚨 **Invalid Submission Detected**\nPlease submit a valid poster, flyer, or graphic design layout. This app is strictly designed for design critique and cannot analyze random photographs or unrelated images.")
        st.stop()

    # Consensus Mechanism for Text Density (Delta = 15%)
    vlm_density = vlm_data.get("estimated_text_density", cv_text_area)
    if abs(cv_text_area - vlm_density) > 15:
        final_text_density = vlm_density # Fallback to AI if math fails on complex background
    else:
        final_text_density = cv_text_area

    with st.spinner("Qwen drafting the final critique…"):
        raw_advice = generate_critique(
            {"balance": balance_str, "text_area": final_text_density,
             "vibrancy": vibrancy, "harmony": harmony},
            vlm_data, hf_key,
        )
        main_rec       = "Improve overall composition."
        detailed_advice = raw_advice
        if "MAIN_REC:" in raw_advice:
            parts          = raw_advice.split("\n\n", 1)
            main_rec       = parts[0].replace("MAIN_REC:", "").replace("**", "").strip()
            detailed_advice = parts[1] if len(parts) > 1 else "See main recommendation."

    # ── Results dashboard ────────────────────────────────────────────────────
    st.divider()
    col_img, col_data, col_color = st.columns([1.2, 1, 1])

    # ── Left column: annotated image ─────────────────────────────────────────
    with col_img:
        st.image(annotated_img, channels="BGR", use_container_width=True)

    # ── Middle column: metrics ────────────────────────────────────────────────
    with col_data:
        # 1. Top Recommendation (Poster-Specific via VLM Top Mistake)
        st.markdown(
            f'<div class="metric-card" style="border: 1px solid #4facfe; background-color: #0b1a26;">'
            f'<div class="metric-title" style="color: #4facfe;">💡 Top Recommendation</div>'
            f'<div style="font-size: 16px; font-weight: bold; color: white;">{main_rec}</div></div>',
            unsafe_allow_html=True,
        )

        # 2. Detected Aesthetic
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Detected Aesthetic</div>'
            f'<div class="metric-value" style="color: #4facfe;">{vlm_data.get("aesthetic", "Unknown")}</div></div>',
            unsafe_allow_html=True,
        )

        # 3. Focal Point
        has_anchor  = vlm_data.get("has_focal_point", True)
        anchor_icon  = "✅" if has_anchor else "❌"
        anchor_color = "#4facfe" if has_anchor else "#ff4b4b"
        anchor_title = "Clear Anchor Detected" if has_anchor else "Missing Focal Point"
        st.markdown(
            f'<div class="metric-card" style="border-left: 4px solid {anchor_color};">'
            f'<div class="metric-title">{anchor_icon} {anchor_title}</div>'
            f'<div style="font-size: 14px; color: #DDD;">{vlm_data.get("focal_point", "None")}</div></div>',
            unsafe_allow_html=True,
        )

        # 4. Visual Hierarchy Gauge
        h_score = vlm_data.get("hierarchy_score", 5)
        h_color = "#ff4b4b" if h_score <= 4 else ("#faca2b" if h_score <= 7 else "#4facfe")
        h_label = "Flat / Monotonous" if h_score <= 4 else ("Moderate Variety" if h_score <= 7 else "Dynamic / Strong")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Visual Hierarchy Gauge</div>'
            f'<div class="metric-value" style="color: {h_color};">{h_label} ({h_score}/10)</div>'
            f'<div style="width:100%;background:#333;border-radius:5px;height:8px;margin-top:8px;">'
            f'<div style="width:{h_score*10}%;background:{h_color};height:8px;border-radius:5px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # 5. Text Density / Clutter Alert
        is_cluttered = final_text_density > 25 and h_score <= 5
        d_border    = "border:1px solid #ff4b4b;background:#2b1111;" if is_cluttered else ""
        d_t_color   = "#ff4b4b" if is_cluttered else "#AAAAAA"
        d_icon      = "⚠️ CLUTTER ALERT" if is_cluttered else "TEXT DENSITY"
        d_label     = ("Overcrowded / Lacks Breathing Room" if is_cluttered
                       else ("Heavy / Editorial" if final_text_density > 30 else "Balanced / Spacious"))
        d_bar_color = "#ff4b4b" if is_cluttered else "#4facfe"
        st.markdown(
            f'<div class="metric-card" style="{d_border}">'
            f'<div class="metric-title" style="color:{d_t_color};">{d_icon}</div>'
            f'<div class="metric-value">{d_label} ({final_text_density}%)</div>'
            f'<div style="width:100%;background:#333;border-radius:5px;height:8px;margin-top:8px;">'
            f'<div style="width:{min(final_text_density,100)}%;background:{d_bar_color};height:8px;border-radius:5px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # 6. Composition Status
        is_balanced = balance_str == "Balanced"
        bal_color   = "#4facfe" if is_balanced else "#ff4b4b"
        bal_icon    = "✅" if is_balanced else "⚠️"
        bal_html    = balance_str.replace("\n", "<br>")
        st.markdown(
            f'<div class="metric-card" style="border-left:4px solid {bal_color};">'
            f'<div class="metric-title">Composition Status</div>'
            f'<div class="metric-value" style="color:{bal_color};font-size:13px;line-height:1.8;">'
            f'{bal_icon} {bal_html}</div></div>',
            unsafe_allow_html=True,
        )

        # 7. Typography Assessment
        font_str   = vlm_data.get("font_suitability", "Unknown")
        font_color = "#4facfe" if "Good" in font_str else "#faca2b"
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Typography Assessment</div>'
            f'<div class="metric-value" style="color:{font_color};">{font_str}</div></div>',
            unsafe_allow_html=True,
        )

        # 8. Dynamic Hard Critique Block (Only shows if amateur or AI-generated)
        is_amateur = vlm_data.get("is_amateur_or_ai", False)
        amateur_critique = vlm_data.get("amateur_critique")
        if is_amateur and amateur_critique:
            st.markdown(
                f'<div class="metric-card" style="border: 1px solid #ff4b4b; background-color: #1a0808;">'
                f'<div class="metric-title" style="color: #ff4b4b;">🛑 Hard Critique (Amateur / AI Detected)</div>'
                f'<div style="font-size: 14px; font-weight: 500; color: #ffcccc; line-height: 1.5;">{amateur_critique}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Right column: colour analysis ─────────────────────────────────────────
    with col_color:
        # Visual Temperature
        temp_label = "Warm Toned" if temp > 55 else ("Cool Toned" if temp < 45 else "Neutral Toned")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Visual Temperature</div>'
            f'<div class="metric-value">{temp_label}</div>'
            f'<div style="width:100%;height:12px;border-radius:6px;'
            f'background:linear-gradient(to right,#2b5876 0%,#4e4376 30%,#b06ab3 50%,#f77062 80%,#fe5196 100%);'
            f'position:relative;margin-top:15px;">'
            f'<div style="position:absolute;left:calc({temp}% - 7px);top:-10px;color:white;font-size:16px;">▼</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Dominant Background
        if bg_color:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Dominant Background ({bg_color["percentage"]}%)</div>'
                f'<div style="width:100%;height:20px;background:{bg_color["hex"]};'
                f'border-radius:5px;border:1px solid #555;"></div></div>',
                unsafe_allow_html=True,
            )

        # Foreground Palette Bar
        bar_html    = "".join(
            f'<div style="width:{c.get("rel_percentage", c["percentage"])}%;background:{c["hex"]};"></div>'
            for c in palette
        )
        labels_html = "<br>".join(
            f'<span style="color:{c["hex"]};">● {c["hex"]} ({c.get("rel_percentage", c["percentage"])}%)</span>'
            for c in palette
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Foreground Palette ({harmony})</div>'
            f'<div style="width:100%;height:18px;display:flex;border-radius:9px;overflow:hidden;'
            f'margin-bottom:10px;border:1px solid #444;">{bar_html}</div>'
            f'<div style="font-size:12px;color:#CCC;">{labels_html}</div></div>',
            unsafe_allow_html=True,
        )

        # Color Wheel
        wheel_palette = core_colors if core_colors else palette
        wheel_dots    = ""
        for c in wheel_palette:
            if c["sat"] > 15:
                rad = math.radians(c["hue"] - 90)
                x, y = 50 + 35 * math.cos(rad), 50 + 35 * math.sin(rad)
                wheel_dots += (
                    f'<div class="color-dot" style="left:{x}%;top:{y}%;background:{c["hex"]};"></div>'
                )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Color Wheel Placement</div>'
            f'<div class="wheel-container"><div class="wheel-inner"></div>{wheel_dots}</div></div>',
            unsafe_allow_html=True,
        )

        # ── Suggested / Approved Palette (VLM-driven) ────────────────────────
        verdict   = vlm_data.get("palette_verdict", "Good Palette")
        suggested = vlm_data.get("suggested_palette")  # dict or None

        if verdict == "Good Palette" or not suggested:
            # Palette approved — just say so
            st.markdown(
                '<div class="metric-card" style="border-left:4px solid #4facfe;">'
                '<div class="metric-title">Suggested Palette</div>'
                '<div style="font-size:20px;margin-bottom:4px;">✅</div>'
                '<div class="metric-value" style="color:#4facfe;">Good Palette</div>'
                '<div style="font-size:12px;color:#666;margin-top:4px;">'
                'Current palette works well for this aesthetic.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Palette needs work — show VLM-suggested swatches with color theory label
            palette_type = suggested.get("type", "Suggested")
            colors_list  = suggested.get("colors", [])
            rationale    = suggested.get("rationale", "")

            if not isinstance(colors_list, list) or len(colors_list) == 0:
                colors_list = [dominant_hex, "#333333", "#555555", "#777777", "#999999"]

            swatches_html = "".join(
                f'<div title="{hx}" style="flex:1;height:28px;background:{hx};'
                f'border-radius:5px;border:1px solid #333;"></div>'
                for hx in colors_list
            )
            hex_labels = "  ".join(
                f'<span style="color:{hx};font-size:10px;">● {hx}</span>'
                for hx in colors_list
            )
            st.markdown(
                f'<div class="metric-card" style="border-left:4px solid #faca2b;">'
                f'<div class="metric-title">Suggested Palette · {palette_type}</div>'
                f'<div style="font-size:12px;color:#faca2b;margin-bottom:10px;">{verdict}</div>'
                f'<div style="display:flex;gap:5px;width:100%;margin-bottom:8px;">{swatches_html}</div>'
                f'<div style="margin-bottom:6px;">{hex_labels}</div>'
                f'<div style="font-size:11px;color:#666;font-style:italic;">{rationale}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Art Director Breakdown ────────────────────────────────────────────────
    st.subheader("Art Director Breakdown")
    st.info(detailed_advice)
import requests
import json
import streamlit as st

# We are using Llama-3-8B-Instruct. It is extremely fast and great at following instructions.
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

def generate_critique(opencv_data):
    """
    Constructs the prompt and calls the Hugging Face API.
    """
    # Grab the key securely from Streamlit
    headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}
    
    # --- PROMPT CONSTRUCTION ---
    # Paragraph 1: Context (and a placeholder note for future reference data)
    context_para = (
        "You are an expert graphic design critic. Your task is to analyze a poster design based on objective technical metrics extracted via computer vision. "
        "Provide a short, structured critique on typography, color harmony, and layout, along with actionable suggestions for improvement. "
        "[Note: Future iterations will include reference datasets here for style matching. For now, rely on standard design principles.]"
    )
    
    # Paragraph 2: Data from OpenCV
    data_para = (
        f"OpenCV Data:\n"
        f"- Number of Text Blocks: {opencv_data['text_blocks']}\n"
        f"- Estimated Font Type: {opencv_data['font_estimate']}\n"
        f"- Text-to-Background Contrast Score: {opencv_data['contrast']}/100 (Higher is more readable)\n"
        f"- Overall Image Tint (Hue): {opencv_data['tint']}\n"
        f"- Overall Image Saturation: {opencv_data['saturation']}%\n"
        f"- Dominant Colors: {opencv_data['color_paragraph']}"
    )
    
    # Combine them into the final prompt sent to the LLM
    final_prompt = f"{context_para}\n\n{data_para}\n\nCritique Output:"
    
    # --- API CALL ---
    payload = {
        "inputs": final_prompt,
        "parameters": {
            "max_new_tokens": 250, # Keep the response punchy
            "temperature": 0.3,    # Low temp so it stays analytical and doesn't hallucinate
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        output = response.json()[0]['generated_text']
        return final_prompt, output
    else:
        return final_prompt, f"Error {response.status_code}: {response.text}"
from huggingface_hub import InferenceClient
import streamlit as st

import os
from dotenv import load_dotenv

# This line loads the environment variables from the .env file
load_dotenv() 

# Now you can safely access the variable
try:
    HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
except KeyError:
    # This error will only show if the variable isn't in .env or isn't set
    st.error("Error: HUGGINGFACE_API_KEY environment variable not found.")
    st.stop()

# --- Author : THAW ZIN ---
# --- To pass the prompt to Hugging Face API and get the response ---
# Note: Please feel free to use this function in your own part of the code 
# Please do not modify this function unless approved by me (Thaw Zin)
def call_huggingface_api(prompt: str) -> str:
    """
    Handles the API call to Hugging Face for text generation.
    """
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
    try:
        output = ""
        stream = client.chat.completions.create(
            # Using a working model instead of the placeholder
            # We can change the model later if needed
            # But Model Access need to be first checked 
            model="mistralai/Mistral-7B-Instruct-v0.2", 
            messages=[{"role": "user", "content": prompt}], # Using chat completion format
            temperature=0.5, # Adjusted for balanced creativity
            max_tokens=2048, # Increased to allow for longer responses
            top_p=0.9, # Adjusted for diversity
            stream=True # Enable streaming
        )
        for chunk in stream:
            if chunk.get("choices"):
                delta_content = chunk["choices"][0].get("delta", {}).get("content", "")
                output += delta_content
        return output.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error occurred while generating the response."

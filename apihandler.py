# --- Author : THAW ZIN HTUN ---
# --- To pass the prompt to Hugging Face API and get the response ---
# Note: Please feel free to use this function in your own part of the code 
# Please do not modify this function unless approved by me (Thaw Zin)

from huggingface_hub import InferenceClient 

import streamlit as st 

import os 

from dotenv import load_dotenv 

load_dotenv() 

# News Fetching Libraries
import feedparser
import urllib.parse


# --- Load API Key from Environment Variable --- 

try: 

    HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"] 

except KeyError: 

    st.error("Error: HUGGINGFACE_API_KEY environment variable not found.") 

    st.stop()  # Stops the app from running further

# please forgive me if appi call fail :3
# I no money , only student :< 
# used free tier huggingface inference api call 
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
        # Stream and concatenate the response
        # This allows for handling large responses without timeouts
        # We assume the response is in chunks with 'choices' and 'delta' keys
        for chunk in stream:
            if chunk.get("choices"):
                delta_content = chunk["choices"][0].get("delta", {}).get("content", "")
                output += delta_content
        return output.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error occurred while generating the response."
    
@st.cache_data(ttl=600) # Cache news for 10 minutes
def fetch_latest_news(query: str, limit: int = 8):
    """Fetches the latest news headlines for a given query from Google News RSS."""
    
    # 1. Encode the query and add "stock" for relevance
    encoded_query = urllib.parse.quote(f"{query} stock")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        # 2. Parse the RSS feed
        feed = feedparser.parse(url)
        if not feed.entries:
            return []
            
        # 3. Sort entries by the 'published_parsed' field (newest first)
        # Uses a fallback tuple (0,...) if published_parsed is missing, making it robust
        sorted_entries = sorted(
            feed.entries, 
            key=lambda entry: entry.published_parsed or (0,0,0,0,0,0,0,0,0), 
            reverse=True
        )
        
        # 4. Return the structured list of dictionaries
        return [
            {
                "Title": entry.title,
                # Use entry.source.title if available, otherwise default to 'Google News'
                "Source": entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News',
                "URL": entry.link
            }
            for entry in sorted_entries[:limit]
        ]
    except Exception as e:
        # 5. Report any errors to the user via st.error
        st.error(f"Error fetching news: {e}")
        return []


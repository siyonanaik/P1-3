# --- Author : THAW ZIN HTUN ---
# --- To pass the prompt to OpenAI API and get the response ---
# Note: Please feel free to use this function in your own part of the code 
# Please do not modify this function unless approved by me (Thaw Zin)
from openai import OpenAI
import streamlit as st 
import os 
from dotenv import load_dotenv 

load_dotenv()  # Load environment variables from a .env file

# News Fetching Libraries
import feedparser
import urllib.parse


# --- Load API Key and model from Environment Variable ---
try: 
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
except KeyError: 
    st.error("Error: OPENAI_API_KEY environment variable not found.") 
    st.stop()  # Stops the app from running further

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_openai_api(prompt: str) -> str:
    """
    Handles the API call to OpenAI for text generation.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "You are an expert financial and technical analyst. Answer clearly and concisely.",
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=2048,
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error occurred while generating the response."


def call_huggingface_api(prompt: str) -> str:
    """Backward-compatible wrapper for older app code."""
    return call_openai_api(prompt)
    
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
            # Sort by published date, newest first
            # using what covered in last python lecture by prof Zheng 
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
            for entry in sorted_entries[:limit]  # Limit results to the requested number
        ]
    except Exception as e:
        # 5. Report any errors to the user via st.error
        st.error(f"Error fetching news: {e}")
        return []  # Return empty list on failure

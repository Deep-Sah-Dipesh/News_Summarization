# All !pip install commands from your original script should be run in your terminal first
# pip install requests beautifulsoup4 newspaper3k google-generativeai transformers torch sentence-transformers nltk tqdm scikit-learn pandas tldextract python-dotenv

import os
import logging
import hashlib
import csv
import torch
import itertools
import time
import requests
import random
import tldextract
from google.api_core import exceptions
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from tqdm import tqdm
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from newspaper import Article, Source # <-- COMPLETED IMPORT

from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
import nltk
import google.generativeai as genai

# --- 1. SETUP AND CONFIGURATION ---

logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- GPU Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- API Key Configuration ---
GEMINI_API_KEYS = [
    # os.getenv("GEMINI_API_KEY_1"),
    # os.getenv("GEMINI_API_KEY_2"),
    # os.getenv("GEMINI_API_KEY_3"),
    # os.getenv("GEMINI_API_KEY_4"),
    # os.getenv("GEMINI_API_KEY_5"),
    # os.getenv("GEMINI_API_KEY_6"),
    # os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9"),
    os.getenv("GEMINI_API_KEY_10"),
    os.getenv("GEMINI_API_KEY_11"),
    os.getenv("GEMINI_API_KEY_12"),
    os.getenv("GEMINI_API_KEY_13"),
    os.getenv("GEMINI_API_KEY_14"),
    os.getenv("GEMINI_API_KEY_15"),
    os.getenv("GEMINI_API_KEY_16"),
    os.getenv("GEMINI_API_KEY_17"),
    os.getenv("GEMINI_API_KEY_18"),
    os.getenv("GEMINI_API_KEY_19"),
    os.getenv("GEMINI_API_KEY_20"),
    os.getenv("GEMINI_API_KEY_21"),
    # ... add all your keys here ...
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]
if not GEMINI_API_KEYS:
    raise ValueError("No Gemini API keys found. Please check your .env file.")

# --- Key Rotation Manager ---
key_cycler = itertools.cycle(GEMINI_API_KEYS)

def get_gemini_model():
    next_key = next(key_cycler)
    print(f"--- Using Gemini API key ending in: ...{next_key[-4:]}")
    genai.configure(api_key=next_key)
    return genai.GenerativeModel("models/gemini-1.5-flash")

# --- Model Loading ---
print("Loading models...")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
print("Models loaded.")

# --- File Paths ---
CSV_PATH = 'scraped_articles.csv'
TOKEN_LIMIT = 512
existing_article_ids = set()
try:
    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'article_id' in row:
                existing_article_ids.add(row['article_id'])
    print(f"Loaded {len(existing_article_ids)} existing article IDs from {CSV_PATH}.")
except FileNotFoundError:
    print(f"'{CSV_PATH}' not found. Starting with an empty set of article IDs.")

# --- 2. HELPER FUNCTIONS ---

def clean_text(text):
    return " ".join(text.strip().split())

def get_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return ""

def compute_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# --- ADDED ROBUST LINK FINDER FUNCTION ---
def get_article_links_from_site(site_url):
    print(f"Attempting to find links for: {site_url}")
    links = set()
    source_domain = tldextract.extract(site_url).registered_domain

    try:
        source = Source(site_url, memoize_articles=False)
        source.build()
        for article_url in source.article_urls():
            article_domain = tldextract.extract(article_url).registered_domain
            if article_domain == source_domain:
                links.add(article_url)
        print(f"Newspaper3k found {len(links)} potential articles.")
    except Exception as e:
        logging.error(f"Newspaper3k source build failed for {site_url}: {str(e)}")

    if not links:
        print("Primary method found no links, trying manual fallback...")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            res = requests.get(site_url, headers=headers, timeout=20)
            soup = BeautifulSoup(res.content, "html.parser")
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(keyword in href for keyword in ['/articles/', '/article/', '/news/']):
                     full_url = requests.compat.urljoin(site_url, href)
                     article_domain = tldextract.extract(full_url).registered_domain
                     if article_domain == source_domain and not full_url.lower().endswith(('.jpg', '.png', '.gif')):
                         links.add(full_url)
        except Exception as e:
            logging.error(f"Manual fallback failed for {site_url}: {str(e)}")
            
    print(f"Found a total of {len(links)} links.")
    return list(links)


def get_article_metadata(article):
    return {
        "author": article.authors,
        "publish_date": str(article.publish_date) if article.publish_date else None,
        "title": article.title,
        "keywords": article.keywords,
        "summary_newspaper": article.summary,
        "meta_description": article.meta_description,
        "meta_keywords": article.meta_keywords,
    }

def summarize_with_bart(text):
    sentences = nltk.sent_tokenize(text)
    inputs = bart_tokenizer(sentences, return_tensors="pt", max_length=TOKEN_LIMIT, truncation=True, padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    summary_ids = bart_model.generate(
        input_ids, attention_mask=attention_mask, max_length=150, min_length=30,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    summaries = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return " ".join(summaries)

def combine_with_gemini(summary):
    prompt = f"Given the following summary, refine it into a comprehensive and concise news article summary in English:\n\n{summary}"
    for attempt in range(len(GEMINI_API_KEYS)):
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            print(f"Key rate limit hit. Retrying with next key... ({attempt + 1}/{len(GEMINI_API_KEYS)})")
            time.sleep(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred with Gemini API: {str(e)}")
            return f"Error: Could not generate summary due to {str(e)}"
    logging.error("All Gemini API keys are rate-limited.")
    return "Error: All API keys have been exhausted."

def translate_to_hindi(text):
    prompt = f"Translate the following English news summary into Hindi:\n\n{text}"
    for attempt in range(len(GEMINI_API_KEYS)):
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            print(f"Key rate limit hit. Retrying with next key... ({attempt + 1}/{len(GEMINI_API_KEYS)})")
            time.sleep(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred with Gemini API: {str(e)}")
            return f"Error: Could not translate summary due to {str(e)}"
    logging.error("All Gemini API keys are rate-limited.")
    return "Error: All API keys have been exhausted."

def append_to_csv(records, csv_path):
    if not records:
        return
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(records[0].keys())
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)

# --- 3. CORE SCRAPING AND PROCESSING LOGIC ---

def scrape_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text or not article.text.strip():
            logging.warning(f"Newspaper3k failed to get content from {url}")
            return None

        content = clean_text(article.text)
        if not content:
            logging.warning(f"Content is empty for {url}")
            return None

        metadata = get_article_metadata(article)
        bart_summary = summarize_with_bart(content)
        combined_summary = combine_with_gemini(bart_summary)
        translated_summary = translate_to_hindi(combined_summary)

        scrape_timestamp = datetime.utcnow().isoformat()
        article_id = compute_hash(url)

        return {
            "article_id": article_id, "source_url": url, "scrape_timestamp": scrape_timestamp,
            "publisher": get_domain(url), "author": metadata["author"], "publish_date": metadata["publish_date"],
            "title_src": metadata["title"], "text_src": content, "summary_src": combined_summary,
            "summary_tgt": translated_summary, "language": "en", "keywords": metadata["keywords"],
            "summary_newspaper": metadata["summary_newspaper"], "meta_description": metadata["meta_description"],
            "meta_keywords": metadata["meta_keywords"], "category": None,
        }
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}", exc_info=True)
        return None

# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    site_sections = [
        # "https://www.bbc.com/news/uk",
        # "https://www.bbc.com/news/world/africa",
        # "https://www.bbc.com/business",
        # "https://www.bbc.com/innovation",
        # "https://www.bbc.com/culture",
        # "https://www.bbc.com/arts",
        # "https://www.bbc.com/travel",
        "https://www.bbc.com/future-planet",
        # "https://www.bbc.com/news/topics/c2vdnvdg6xxt",
        # "https://www.bbc.com/news/war-in-ukraine",
        # "https://www.bbc.com/news/world/latin_america"
        # Add more section URLs as needed
    ]

    for section_url in site_sections:
        article_links = get_article_links_from_site(section_url)
        if not article_links:
            print(f"No links found for {section_url}. Skipping.")
            continue
        
        category = get_domain(section_url) + urlparse(section_url).path.replace('/', '_')

        for url in tqdm(article_links, desc=f"Scraping {category}"):
             # --- ADD THIS DUPLICATE CHECK BLOCK ---
            potential_article_id = compute_hash(url)
            if potential_article_id in existing_article_ids:
                # print(f"Skipping duplicate article ID: {potential_article_id}")
                continue # Skip to the next URL
            # --- END OF NEW BLOCK ---
            
            
            article_data = scrape_url(url)
            if article_data:
                article_data['category'] = category
                append_to_csv([article_data], csv_path=CSV_PATH)
                print(f"\nAppended: {article_data['title_src'][:50]}... from {category}")

    print(f"\n--- DONE ---")
    print(f"Scraping complete. Data saved to {CSV_PATH}")
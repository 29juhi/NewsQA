import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import torch
from urllib.parse import urlparse

# Loading the models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

st.set_page_config(page_title="News Q&A", page_icon="📰", layout="wide")

st.title('📰 News Q&A')
st.write('Enter URLs of news articles and questions to get answers.')

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def fetch_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def embed_articles(articles):
    all_paragraphs = []
    all_embeddings = []
    article_sources = []
    
    for url, article in articles:
        paragraphs = article.split('\n')
        embeddings = model.encode(paragraphs)
        all_paragraphs.extend(paragraphs)
        all_embeddings.extend(embeddings)
        article_sources.extend([url] * len(paragraphs))
        
    return all_paragraphs, np.array(all_embeddings), article_sources

def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Dimension of embeddings
    index.add(embeddings)
    return index

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

def get_answer(question, all_paragraphs, index, article_sources):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=5)
    
    candidate_paragraphs = [all_paragraphs[i] for i in I[0]]
    candidate_sources = [article_sources[i] for i in I[0]]
    
    best_answer = None
    best_source = None
    best_score = 0
    
    for paragraph, source in zip(candidate_paragraphs, candidate_sources):
        result = qa_pipeline(question=question, context=paragraph)
        if result['score'] > best_score:
            best_score = result['score']
            best_answer = result['answer']
            best_source = source
            
    return best_answer, best_source

# Add custom CSS to change the background color
st.markdown("""
    <style>
    .main {
        background-color: #BB9AB1;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #987D9A;
    }
    </style>
    """, unsafe_allow_html=True)

# Layout
st.sidebar.title("User Input")
url1 = st.sidebar.text_input('Enter news article URL 1')
url2 = st.sidebar.text_input('Enter news article URL 2')
url3 = st.sidebar.text_input('Enter news article URL 3')
question = st.sidebar.text_input('Enter your question')

if st.sidebar.button('Get Answer'):
    if any([url1, url2, url3]) and question:
        valid_urls = [url for url in [url1, url2, url3] if is_valid_url(url)]
        if valid_urls:
            with st.spinner('Fetching articles and generating answer...'):
                articles = [(url, fetch_article(url)) for url in valid_urls]
                all_paragraphs, all_embeddings, article_sources = embed_articles(articles)
                index = create_index(all_embeddings)
                answer, source = get_answer(question, all_paragraphs, index, article_sources)
            st.success('Answer generated successfully!')
            st.write('### Answer:', answer)
            st.write('### Source:', source)
        else:
            st.error('Please provide valid URLs.')
    else:
        st.error('Please provide both URLs and a question.')
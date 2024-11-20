📰 News Q&A Streamlit App
This project provides a simple and interactive way to ask questions based on the content of news articles. By inputting URLs of news articles and a question, the app fetches the articles, extracts their content, and uses natural language processing (NLP) to provide an answer to the user's query. The app leverages pre-trained models from Hugging Face and Sentence Transformers to understand and answer the question contextually.

Features
URL Input: Users can input up to 3 news article URLs.
Question Input: Users can type a question related to the news articles.
Answer Extraction: The app fetches content from the URLs, processes it, and uses a question-answering model to return the most relevant answer.
Contextual Answering: By leveraging both sentence embeddings (using Sentence Transformers) and a question-answering pipeline (using Hugging Face’s DistilBERT model), the app provides contextually accurate answers.
User-Friendly Interface: Built using Streamlit with custom styles for an enhanced user experience.
Requirements
To run this application, you will need the following Python packages:

streamlit: For building the web interface.
requests: For fetching content from URLs.
beautifulsoup4: For parsing HTML content and extracting text.
sentence-transformers: For generating sentence embeddings.
faiss: For efficient similarity search.
transformers: For using pre-trained NLP models.
torch: PyTorch framework (for model inference).
Install the dependencies
bash
Copy code
pip install streamlit requests beautifulsoup4 sentence-transformers faiss-cpu transformers torch
How to Run
Clone this repository or download the script.

Ensure all dependencies are installed (as mentioned above).

Run the Streamlit app using the following command in your terminal:

bash
Copy code
streamlit run app.py
Open the app in your browser (usually at http://localhost:8501).

Enter the URLs of the news articles you want to extract information from.

Enter your question in the sidebar and click "Get Answer".

How It Works
Fetch Articles: The app fetches the content from the provided URLs using the requests library and parses it using BeautifulSoup to extract the main text.

Embedding Articles: The articles are split into paragraphs, and each paragraph is embedded using the SentenceTransformer model. This creates a dense vector representation of each paragraph for comparison.

Indexing: The embeddings are indexed using FAISS (Facebook AI Similarity Search), which allows efficient retrieval of similar paragraphs based on a given question.

Question Answering: When a user inputs a question, the app generates an embedding for the question and searches for the most relevant paragraphs from the articles. It then uses a pre-trained DistilBERT model to extract the best answer from the most relevant paragraphs.

Display Results: The app displays the best answer found along with the source article from which it was derived.

Example Usage
Enter news URLs (e.g., from major news websites).
Enter a question about the article (e.g., "What is the main theme of the article?").
Click "Get Answer" and wait for the response.

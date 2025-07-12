import requests
from bs4 import BeautifulSoup
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    return ' '.join([word for word in tokens if word not in stopwords and len(word) > 2])

# Web scraping
def fetch_text_from_url(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.text for p in paragraphs)
        return preprocess(text)
    except:
        return ""

# Sample URLs (replace with real ones)
urls = [
    'https://en.wikipedia.org/wiki/Natural_language_processing',
    'https://en.wikipedia.org/wiki/Web_scraping',
    'https://en.wikipedia.org/wiki/Machine_learning'
]

# Fetch and preprocess
documents = [fetch_text_from_url(url) for url in urls]
df = pd.DataFrame({'url': urls, 'text': documents})

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Simulated classification data (random labels for demo)
df['label'] = [1, 0, 1]  # 1 = Relevant, 0 = Irrelevant

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.3, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, preds))

# Simple summarizer (top N sentences with highest TF-IDF score)
def summarize(text, n=3):
    sentences = nltk.sent_tokenize(text)
    tfidf = TfidfVectorizer()
    sent_vectors = tfidf.fit_transform(sentences)
    scores = sent_vectors.sum(axis=1).flatten().tolist()[0]
    ranked_sentences = [sentences[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    return ' '.join(ranked_sentences[:n])

print("\n--- Sample Summary ---")
print(summarize(df['text'][0]))

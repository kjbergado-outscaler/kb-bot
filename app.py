# app.py
import os
import glob
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

# -----------------------------
# CONFIGURATION
# -----------------------------
KB_CSV = "KB_Sheet.csv"        # Your CSV file
HTML_FOLDER = "kb_html_files"  # Folder with HTML KB files
EMBED_FILE = "kb_embeddings.npy"
SUMM_FILE = "kb_summaries.npy"
TOP_K = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMM_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "google/flan-t5-small"

# -----------------------------
# LOAD KB DATA
# -----------------------------
print("📄 Loading KB CSV...")
kb_sheet = pd.read_csv(KB_CSV)

kb_data = []
texts = []

for _, row in kb_sheet.iterrows():
    kb_id = str(row['Export Import Id'])
    matches = glob.glob(os.path.join(HTML_FOLDER, f"*{kb_id}*.html"))
    if matches:
        with open(matches[0], 'r', encoding='utf-8') as f:
            html = f.read()
        text_content = BeautifulSoup(html, 'html.parser').get_text(separator=" ", strip=True)
        kb_data.append({
            "category": row.get('CATEGORY', ''),
            "subcategory": row.get('SUBCATEGORY', ''),
            "title": row.get('Title', ''),
            "content": text_content,
            "kb_link": row.get('KB LINK', ''),
            "last_modified": row.get('Modified On', '')
        })
        texts.append(text_content)
    else:
        print(f"⚠️ No HTML file found for KB ID: {kb_id}")

if len(kb_data) == 0:
    raise ValueError("⚠️ No KB content loaded. Check your kb_html_files folder.")

print(f"✅ Loaded {len(kb_data)} KB articles.")

# -----------------------------
# EMBEDDINGS
# -----------------------------
print("🧠 Loading embeddings...")
model = SentenceTransformer(MODEL_NAME)
if os.path.exists(EMBED_FILE):
    embeddings = np.load(EMBED_FILE)
    print("✅ Loaded precomputed embeddings.")
else:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMBED_FILE, embeddings)
    print("✅ Computed and saved embeddings.")

# FAISS index
dimension = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print("✅ FAISS index ready.")

# -----------------------------
# SUMMARIES
# -----------------------------
print("📝 Loading summaries...")
summarizer = pipeline("summarization", model=SUMM_MODEL)
if os.path.exists(SUMM_FILE):
    short_summaries = np.load(SUMM_FILE, allow_pickle=True)
    print("✅ Loaded precomputed summaries.")
else:
    short_summaries = []
    for article in kb_data:
        try:
            summary = summarizer(article['content'], max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        except Exception:
            summary = article['content'][:100]
        short_summaries.append(summary)
    np.save(SUMM_FILE, short_summaries)
    print("✅ Computed short summaries.")

# -----------------------------
# QA MODEL
# -----------------------------
qa_model = pipeline("text2text-generation", model=QA_MODEL)

def answer_question_fast(query, top_k=TOP_K):
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)

    context = " ".join([short_summaries[i] for i in I[0]])
    answer = qa_model(
        f"Answer the question based on the context:\nContext: {context}\nQuestion: {query}",
        max_new_tokens=150
    )[0]['generated_text']

    recommendations = []
    for i in I[0]:
        recommendations.append({
            "category": kb_data[i]['category'],
            "subcategory": kb_data[i]['subcategory'],
            "title": kb_data[i]['title'],
            "snippet": short_summaries[i],
            "kb_link": kb_data[i]['kb_link'],
            "last_modified": kb_data[i]['last_modified']
        })

    disclaimer = "⚠️ Disclaimer: Always double-check KB for updates."
    return disclaimer, answer, recommendations

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    html = """
    <h2>📚 KB Bot</h2>
    <p>Use POST /chat with JSON: {"query":"your question"}</p>
    """
    return render_template_string(html)

# Chat route
@app.route("/chat", methods=["POST"])
def chat():
    req = request.get_json()
    query = req.get("query", "")
    if not query:
        return jsonify({"error": "No query provided."}), 400

    disclaimer, answer, recs = answer_question_fast(query)

    # Format HTML output
    html_output = f"<p>{disclaimer}</p><h3>💡 Closest Answer:</h3><p>{answer}</p>"
    html_output += "<h3>📌 Recommended KB Topics:</h3><ul>"
    for r in recs:
        html_output += f"<li><strong>{r['title']}</strong> ({r['kb_link']})<br>Category: {r['category']} | Subcategory: {r['subcategory']}<br>Preview: {r['snippet']}<br>Last Modified: {r['last_modified']}</li>"
    html_output += "</ul>"

    return html_output

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
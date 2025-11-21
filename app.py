# app.py
import os
import glob
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from flask import Flask, request, jsonify

# -----------------------------
# CONFIGURATION
# -----------------------------
KB_CSV = "KB_Sheet.csv"           # Your CSV in the project folder
HTML_FOLDER = "kb_html_files"     # Folder containing HTML files
TOP_K = 3                         # Top 3 recommendations
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# LOAD KB DATA
# -----------------------------
print("📄 Loading KB CSV...")
kb_sheet = pd.read_csv(KB_CSV)

kb_data = []
texts = []

for _, row in kb_sheet.iterrows():
    kb_id = str(row['Export Import Id'])
    # Look for matching HTML file
    matches = glob.glob(os.path.join(HTML_FOLDER, f"*{kb_id}*.html"))
    if matches:
        with open(matches[0], 'r', encoding='utf-8') as f:
            html = f.read()
        text_content = BeautifulSoup(html, 'html.parser').get_text(separator=" ", strip=True)
        kb_entry = {
            "category": row.get('CATEGORY', ''),
            "subcategory": row.get('SUBCATEGORY', ''),
            "title": row.get('Title', ''),
            "content": text_content,
            "kb_link": row.get('KB LINK', ''),
            "last_modified": row.get('Modified On', '')
        }
        kb_data.append(kb_entry)
        texts.append(text_content)
    else:
        print(f"⚠️ No HTML file found for KB ID: {kb_id}")

if len(kb_data) == 0:
    raise ValueError("⚠️ No KB content loaded. Check your kb_html_files folder.")

print(f"✅ Loaded {len(kb_data)} KB articles.")

# -----------------------------
# EMBEDDINGS
# -----------------------------
print("🧠 Creating embeddings...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ Embeddings and FAISS index ready.")

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

def answer_question_fast(query, top_k=TOP_K):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)

    recommendations = []
    for idx in I[0]:
        kb = kb_data[idx]
        snippet = kb['content'][:250] + "..." if len(kb['content']) > 250 else kb['content']
        recommendations.append({
            "title": kb['title'],
            "category": kb['category'],
            "subcategory": kb['subcategory'],
            "snippet": snippet,
            "kb_link": kb['kb_link'],
            "last_modified": kb['last_modified']
        })

    disclaimer = "⚠️ This answer is AI-assisted. Please verify against the official KB."
    answer_text = recommendations[0]['snippet'] if recommendations else "No relevant KB found."
    return disclaimer, answer_text, recommendations

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided."}), 400

    disclaimer, answer, recs = answer_question_fast(query)
    return jsonify({
        "disclaimer": disclaimer,
        "answer": answer,
        "recommendations": recs
    })

# Root route
@app.route("/", methods=["GET"])
def home():
    return "KB Bot is running. Use POST /ask with {'query':'your question'}"

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
Run on your own computer (internet needed once to download MiniLM):
  pip install -r requirements.txt sentence-transformers scikit-learn reportlab pillow
  python gen_docs.py            # recreates synthetic PDFs + ground_truth.json
  python run_experiments.py PATH_TO_YOUR_CLONED_REPO
The script writes results_<backend>.json. It prints "actually used"; it must say
chroma_onnx or sentence_transformers, NOT hash_vectorizer. Send me that JSON and I fill the MiniLM column.
fixed_files/: corrected vector_store.py (no duplicate chunks on restart) and embedder.py (stable hashing).

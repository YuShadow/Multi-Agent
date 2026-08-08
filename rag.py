"""
RAG utilities for the Document Agent.

Unlike LangGraph.py's original approach (which expected a local folder of
.txt files that were never committed to the repo), this module builds its
index at runtime from whatever PDF the user uploads in the Streamlit app.
No local files, no hardcoded paths - the corpus is always exactly one
user-supplied PDF.
"""
from typing import List

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def extract_pdf_text(file) -> str:
    """Extract text from an uploaded PDF (file-like object, e.g. from
    st.file_uploader). Pages with no extractable text (e.g. scanned
    images with no OCR layer) are silently skipped."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping fixed-size chunks. Same scheme used in
    the original LangGraph.py prototype, kept consistent here."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class PDFIndex:
    """In-memory FAISS index over a single uploaded PDF's chunks.

    A fresh instance is built per uploaded file (see main.py) rather than
    accumulating documents across uploads, since the goal is "answer from
    this PDF," not a persistent multi-document corpus.
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.chunks: List[str] = []
        self.index = None

    def build(self, chunks: List[str]) -> None:
        self.chunks = chunks
        if not chunks:
            self.index = None
            return

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Return the top_k most relevant chunks for a query. Returns an
        empty list if the index hasn't been built or has no chunks -
        callers should handle that gracefully rather than assume results."""
        if self.index is None or not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        _, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]
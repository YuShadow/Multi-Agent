from typing import List

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
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
        if self.index is None or not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        _, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]

# Lecture QA System using Finetuned Transformer

A **Retrieval-Augmented Generation (RAG)** system that lets students ask natural-language questions about their lecture PDFs and get accurate, cited answers instantly.

Built for AIE343 — Machine Learning for Text Mining at Galala University.

---

## What it does

- Upload lecture PDFs and ask any question about the content
- Get answers grounded in the actual lecture material with page citations
- Ask follow-up questions naturally — the system remembers the conversation
- Generate study notes and interactive multiple-choice quizzes from your lectures
- Save and reload chat sessions across browser refreshes

---

## How it works

Uploaded PDFs are split into overlapping chunks, embedded using a state-of-the-art retrieval model, and indexed for fast search. When you ask a question, the most relevant chunks are retrieved, reranked for precision, and passed to a fast LLM that generates a grounded answer.

| Component | Choice |
|---|---|
| Embedding | BAAI/bge-large-en-v1.5 |
| Vector search | FAISS |
| Reranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Llama-3.1-8B via Groq |
| UI | Streamlit |

---
S |

**Supervisor:** Dr. Manar El-Shazly — Galala University, 2024–2025

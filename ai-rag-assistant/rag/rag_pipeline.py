from rag.embedding import embed_text
from rag.vector_store import VectorStore
import requests
import os
from dotenv import load_dotenv

# load .env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

store = VectorStore()


def load_documents():

    with open("data/documents.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()

    docs = [d.strip() for d in docs if d.strip()]

    embeddings = embed_text(docs)

    store.add(embeddings, docs)


load_documents()


def generate_answer(context, question):

    prompt = f"""
You are an AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    result = response.json()

    # safe error handling
    if "choices" not in result:
        return f"API Error: {result}"

    return result["choices"][0]["message"]["content"]


def ask_question(question):

    q_embedding = embed_text([question])[0]

    context_docs = store.search(q_embedding)

    context = "\n".join(context_docs)

    answer = generate_answer(context, question)

    return answer, context_docs
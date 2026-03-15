import numpy as np

class VectorStore:

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, embeddings, texts):
        for emb, text in zip(embeddings, texts):
            self.vectors.append(emb)
            self.texts.append(text)

    def search(self, query_embedding, top_k=2):

        similarities = []

        for vector in self.vectors:

            sim = np.dot(query_embedding, vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vector)
            )

            similarities.append(sim)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [self.texts[i] for i in top_indices]

        return results
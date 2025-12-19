from langchain_ollama import OllamaEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Define the documents
documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]
document_embeddings = embeddings.embed_documents(documents)


query = "A cat is sitting on a mat."
query_embedding = embeddings.embed_query(query)

# Calculate similarity scores
similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Find the most similar document
most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Most similar document to the query '{query}':")
print(most_similar_document)
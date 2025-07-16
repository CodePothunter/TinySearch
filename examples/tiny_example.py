"""
Tiny example for TinySearch
The most minimal example possible to demonstrate core functionality
Using Qwen3-Embedding-0.6B for efficient embedding performance
"""

# Minimal imports
from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from pathlib import Path

# Helper function to format instructions for queries
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# --- Step 1: Extract text from files ---
adapter = TextAdapter()
texts = []
file_texts = adapter.extract("example_data/sample.txt")
texts.extend(file_texts)
print(f"Extracted {len(texts)} text segments")

# --- Step 2: Split text into chunks ---
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split(texts)
print(f"Created {len(chunks)} text chunks")

# --- Step 3: Generate embeddings using Qwen3-Embedding-0.6B ---
embedder = HuggingFaceEmbedder(device="cpu")
chunk_texts = [chunk.text for chunk in chunks]
vectors = embedder.embed(chunk_texts)
print(f"Generated {len(vectors)} embedding vectors with dimension {len(vectors[0]) if vectors else 0}")

# --- Step 4: Create and build index ---
indexer = FAISSIndexer(use_gpu=False)
indexer.build(vectors, chunks)
print("Built FAISS index")

# --- Step 5: Query the index with instruction ---
# Define instruction for the query to improve performance
task = "Given a web search query, retrieve relevant passages that answer the query"
query_text = "What is vector search?"
instructed_query = get_detailed_instruct(task, query_text)

print(f"\nInstructed query: {instructed_query}")
query_vector = embedder.embed([instructed_query])[0]
results = indexer.search(query_vector, top_k=3)

# --- Step 6: Display results ---
print("\nSearch Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['text'][:100]}... (Score: {result['score']:.2f})")

# Optional: Save the index
index_path = Path("tiny_index.faiss")
indexer.save(index_path)
print(f"\nIndex saved to {index_path}")
print("\nThis example demonstrates TinySearch using Qwen3-Embedding-8B, the latest state-of-the-art embedding model from Qwen.") 
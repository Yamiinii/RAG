import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_KEY")

if not api_key:
    raise ValueError("API Key not found. Check your .env file.")

import google.generativeai as genai
genai.configure(api_key=api_key)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function  # ✅ Correct way to pass the embedding function
)

print("Done initialising")

def query_gemini(question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return response.text

question = "What are the latest investments in supply chain startups?"
before_rag = query_gemini(question)
print("\n=== Before RAG ===\n", before_rag)


def load_doc_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents=[]
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path,filename),"r",encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./new-articles"
documents = load_doc_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

def get_embedding(text):
    embedding = embedding_model.encode(text) 
    return embedding

for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_embedding(doc["text"])

# print(doc["embedding"])

for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

def query_documents(question, n_results=2):
    print("==== Retrieving relevant facts ====")
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    
    if not relevant_chunks:
        return "No relevant data found."
    
    return "\n".join(relevant_chunks)

def generate_answer(question):
    context = query_documents(question)
    
    if context == "No relevant data found.":
        return "I don't have enough information to answer that."
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "

    print("==== Generating response using Gemini ====")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    return response.text

question = "What are the latest investments in supply chain startups?"
answer = generate_answer(question)
print("\nFinal Answer:", answer)
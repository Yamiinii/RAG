# RAG Implementation with Gemini-1.5-Flash

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using **Gemini-1.5-Flash**, **ChromaDB**, and **SentenceTransformers**. The system enhances the accuracy of responses by retrieving relevant data from stored documents before generating an answer. The pipeline involves:

- Querying Gemini before retrieval (Baseline response)
- Loading and chunking text data from a directory
- Generating embeddings for document chunks using **SentenceTransformers**
- Storing and retrieving document embeddings using **ChromaDB**
- Using retrieved context to generate an enhanced answer with **Gemini-1.5-Flash**
- Comparing responses before and after RAG

## Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies from `requirements.txt`

### Installation
1. Clone the repository:
   ```sh
   git clone <repo-link>
   cd <repo-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Create a `.env` file
   - Add your **Gemini API Key**:
     ```env
     GEMINI_KEY=your_api_key_here
     ```

## How It Works

### Step 1: Query Gemini (Baseline Response)
Before utilizing retrieval, we ask **Gemini-1.5-Flash** a question directly:
```python
question = "What are the latest investments in supply chain startups?"
before_rag = query_gemini(question)
print("\n=== Before RAG ===\n", before_rag)
```

### Step 2: Load and Process Documents
- Load text documents from `./new-articles/`
- Split long documents into smaller chunks for efficient embedding
- Generate embeddings using **SentenceTransformers** (`all-MiniLM-L6-v2`)
- Store chunked documents and embeddings in **ChromaDB**

### Step 3: Query ChromaDB and Enhance Answer
When a question is asked:
1. Retrieve relevant document chunks from **ChromaDB**.
2. Pass the retrieved context to **Gemini-1.5-Flash** for response generation.
3. Compare the answer with the original baseline response.

### Sample Workflow
```python
question = "What are the latest investments in supply chain startups?"
answer = generate_answer(question)
print("\nFinal Answer:", answer)
```

## Example Articles Processed
The system processes text articles stored in `./new-articles/`. Sample articles:
1. **Article 1**: "Recent AI Investments in Logistics"
2. **Article 2**: "Supply Chain Startups: Growth and Challenges"
3. **Article 3**: "Tech Innovations in Global Shipping"

## Results Comparison
| Query | Before RAG (Direct Gemini) | After RAG (With Retrieval) |
|--------|--------------------------|--------------------------|
| **"What are the latest investments in supply chain startups?"** | *Pinpointing the very latest investments is difficult because funding rounds are announced constantly...* *(Generic response covering trends like AI, automation, sustainability, etc.)* | *Pando, a fulfillment management technology startup, recently raised $30 million in a Series B funding round, bringing its total funding to $45 million...* *(Factual answer using retrieved documents.)* |

## Example Article Used for Retrieval
One of the documents used discusses **Pando**, a supply chain startup that raised **$30M** in a Series B round. This factual data is retrieved by **ChromaDB** and provided to Gemini for better responses.

## Future Improvements
- Integrate **more advanced retrieval techniques** (e.g., BM25, Hybrid Search)
- Expand document storage with **real-time news updates**
- Fine-tune retrieval **for specific domains** (e.g., finance, healthcare)
- Optimize response generation **for factual accuracy**

## Author
Developed by **[Your Name]**


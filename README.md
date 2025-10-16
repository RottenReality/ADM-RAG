# 📊 AgroChain RAG API: Agribusiness Traceability Analysis

## Project Overview

The AgroChain RAG API is a specialized solution implementing a Retrieval-Augmented Generation (RAG) system to enable natural language querying of complex agribusiness supply chain data (agrichains).

Its core mission is to transform deeply nested JSON traceability documents—containing key metrics such as duration and performance status across various production stages—into an intelligent, searchable knowledge base. This allows non-technical users, such as analysts and supply chain managers, to gain insights by simply asking questions, without needing to write database queries.

---

## Key Features

- **Semantic Querying:** Semantic Querying: Interact with traceability data using natural, conversational Spanish or English (depending on the LLM).
- **Deep Context Retrieval:** Embedding-based retrieval ensures relevant answers to complex questions about stages, participants, and statuses.
- **Designed for Analytics:** Prepared to support analytical questions such as comparing real vs. estimated durations or identifying costly stages.
- **RESTful API:** Built on FastAPI, exposing a simple POST endpoint for easy integration into dashboards or reporting tools.
- **Persistent Indexing:** Persistent Indexing: Uses ChromaDB for storing document embeddings, allowing for fast and repeated queries without re-indexing.

---

## 🛠️ Technology Stack

| Component       | Role                  | Details                                   |
|-----------------|----------------------|-------------------------------------------|
| Backend API     | Web Framework         | FastAPI & Uvicorn                        |
| LLM (Chat)      | Reasoning & Generation| Google Gemini 2.5 Flash                   |
| RAG Framework   | Orchestration         | LlamaIndex (Core)                        |
| Vector Database | Persistent Storage    | ChromaDB                                 |
| Embeddings      | Text-to-Vector        | HuggingFaceEmbedding (BAAI/bge-small-en-v1.5) |
| Configuration   | Environment Secrets   | python-dotenv                            |

---

## ⚙️ Data Handling and RAG Architecture

- **Data Flattening:** The nested JSON structure (`agrocadenas` > `etapas`) is iteratively traversed to extract every production stage.
- **Node Creation:** Each stage (etapa) is converted into a `TextNode`, preserving its content and structure.
- **Metadata Enrichment:** Every node is tagged with key metadata—such as the agrichain name (agrocadena) and participant status (estado_participante)—to enable future filtering and contextual understanding.
- **Indexing:** The nodes are embedded using the local `BAAI/bge-small-en-v1.5` model via the LlamaIndex `VectorStoreIndex` interface and stored persistently in `ChromaDB`.
- **Query Engine:** Configured with `similarity_top_k=10` to retrieve a broad and context-rich set of relevant nodes, ensuring that responses incorporate sufficient background across multiple stages of the supply chain.

---

## 🚀 Setup and Local Execution

### Prerequisites

- Python 3.9+
- Your Gemini API Key
- The data file `agrocadenas.json` in the project root

---

### Step 1: Clone the Repository and Install Dependencies

Clone your repository (assuming you have pushed it)
git clone <YOUR_REPO_URL>
cd agrochain-rag-api

Install required Python packages
pip install fastapi uvicorn llama-index-core llama-index-llms-gemini llama-index-vector-stores-chroma chromadb python-dotenv

---

### Step 2: Configure Environment Variables

Create a file named `.env` in the project root directory and add your Gemini API Key:

.env file:
`GEMINI_API_KEY="YOUR_API_KEY_HERE"`

---

### Step 3: Run the API

Execute the main Python file. This command will also initialize the ChromaDB index (`chroma_db/`) if it doesn't already exist.

python main.py

The API will be running locally at [http://0.0.0.0:8000](http://0.0.0.0:8000).

---

### Step 4: Test the Endpoint

Access the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the `/consulta/` endpoint for testing.

**Example POST Request Body:**
```
{
    "pregunta": "Review all the stages where the participant has an inactive status. From that list, identify and enumerate the Chain, Stage, and participant, and list these details."
}
```

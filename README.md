# 📊 AgroChain RAG API: Agribusiness Traceability Analysis

## Project Overview

The AgroChain RAG API is a specialized solution implementing a Retrieval-Augmented Generation (RAG) system to enable sophisticated, natural language querying of complex agribusiness supply chain data (agrichains).  
The core mission is to transform deeply nested JSON traceability documents—which contain critical metrics like duration and performance status across various production stages—into an intelligent knowledge base.
This allows non-technical users, such as analysts and supply chain managers, to gain instantaneous insights by simply asking questions, eliminating the need for complex database queries.

---

## Key Features

- **Semantic Querying:** Interact with traceability data using natural, conversational Spanish (or English, depending on the query).
- **Deep Context Retrieval:** The RAG architecture is optimized for nested data, ensuring high relevance when answering complex questions about specific stages, participants, and statuses (e.g., identifying the cost of `"Alert"` status stages).
- **Time and Cost Analytics:** Capable of performing calculations on the fly, such as comparing real duration vs. estimated duration (handling unit conversion from days to hours).
- **RESTful API:** Built on FastAPI to provide a scalable and simple POST endpoint for seamless integration into dashboards, reporting tools, and other applications.
- **Persistent Indexing:** Uses ChromaDB to store the document embeddings, allowing for fast, repeated queries without re-indexing the entire dataset.

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

- **Data Flattening:** The nested JSON structure (`agrocadenas` > `etapas`) is traversed.
- **Node Creation:** A `TextNode` is created for every individual stage (etapa) within each agrichain.
- **Metadata Enrichment:** Each node is tagged with essential metadata, including the agrocadena name and `estado_participante` (participant status), crucial for filtering and conditional queries.
- **Indexing:** The nodes are converted into vectors using the `BAAI/bge-small-en-v1.5` model and stored in ChromaDB.
- **Query Engine:** The query engine is configured with a high `similarity_top_k=10` to ensure that enough context (up to 10 relevant stages) is retrieved to answer complex analytical and comparison questions accurately.

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

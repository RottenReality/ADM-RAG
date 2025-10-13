import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings # <-- Agregado 'Settings'
from llama_index.core.schema import TextNode
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- CONFIGURACIÓN Y CONSTANTES ---
load_dotenv()
JSON_PATH = 'agrocadenas.json'
CHROMA_PATH = './chroma_db'
COLLECTION_NAME = "agrocadenas_index"

# --- CARGA Y SEGMENTACIÓN DE DATOS (CHUNK & NODES) ---

def cargar_datos(ruta_json: str):
    """Carga los datos del JSON."""
    with open(ruta_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def crear_nodos_rag(datos: dict) -> list[TextNode]:
    """Aplaza el JSON y crea Nodos (Chunks) con metadatos."""
    nodes = []
    for agro_cadena in datos.get('agrocadenas', []):
        nombre_ac = agro_cadena.get('nombre')
        for etapa in agro_cadena.get('etapas', []):
            texto_chunk = json.dumps(etapa, ensure_ascii=False, indent=2)
            node = TextNode(
                text=texto_chunk,
                metadata={
                    "agrocadena": nombre_ac,
                    "etapa_nombre": etapa.get('nombre'),
                    "estado_participante": etapa.get('participante', {}).get('estado')
                }
            )
            nodes.append(node)
    return nodes

datos_agrocadenas = cargar_datos(JSON_PATH)
nodos_rag = crear_nodos_rag(datos_agrocadenas)
print(f"Datos cargados: {len(nodos_rag)} nodos listos.")

# --- INICIALIZACIÓN DE MODELOS E ÍNDICE ---

# Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 

# LLM de Gemini
llm = Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")) 

Settings.llm = llm
Settings.embed_model = embed_model

# ChromaDB
print("Inicializando ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Usamos el método corregido, sin pasar el embed_model a Chroma
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Crear el Índice RAG
print("Creando índice vectorial...")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex(
    nodes=nodos_rag, 
    embed_model=embed_model, 
    llm=llm,
    vector_store=vector_store
)
print("¡Indexación completa con Gemini y ChromaDB!")

# Motor de consultas RAG
query_engine = index.as_query_engine(similarity_top_k=10)

# --- API ---

# Define la estructura de la solicitud
class Consulta(BaseModel):
    pregunta: str

app = FastAPI(
    title="API RAG Agrocadenas con Gemini",
    description="Backend para consultar datos de trazabilidad y optimización."
)

def consultar_rag(pregunta: str) -> str:
    """Función que maneja la consulta al motor RAG usando el LLM de Gemini."""
    try:
        respuesta = query_engine.query(pregunta)
        return str(respuesta)
    except Exception as e:
        return f"Error al procesar la consulta: {e}"

@app.post("/consulta/")
async def consulta_rag_api(consulta: Consulta):
    """Endpoint para recibir una pregunta y devolver la respuesta del RAG."""
    respuesta = consultar_rag(consulta.pregunta)
    return {"pregunta": consulta.pregunta, "respuesta": respuesta}

# --- EJECUCIÓN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

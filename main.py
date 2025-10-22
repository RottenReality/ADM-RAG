import json
import os
import re
from typing import Optional, Dict, Any
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
from llama_index.core import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware

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

qa_prompt_template_str = (
    "Eres un asistente experto en análisis de datos de agrocadenas.\n"
    "Contexto de la cadena de suministro:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Basado en el contexto, responde la siguiente pregunta: {query_str}\n\n"
    "--- INSTRUCCIONES ADICIONALES ---\n"
    "1. Primero, escribe una respuesta textual clara y concisa.\n"
    "2. Si la pregunta implica una comparación numérica, una distribución o datos que puedan ser visualizados en un gráfico (barras, torta, etc.), "
    "después de tu respuesta textual, añade un separador especial '[GRAFICO_JSON]' seguido de un bloque de código JSON con los datos para el gráfico.\n"
    "3. El JSON debe ser compatible con Chart.js y tener la siguiente estructura: { 'type': 'bar'|'pie'|'line', 'labels': [...], 'datasets': [{ 'label': '...', 'data': [...] }] }.\n"
    "4. Si la respuesta es puramente textual y no contiene datos para un gráfico, NO incluyas el separador ni el bloque JSON.\n"
)
qa_prompt_template = PromptTemplate(qa_prompt_template_str)

# Motor de consultas RAG
query_engine = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=qa_prompt_template
)

# --- API ---

class GraficoData(BaseModel):
    """Define la estructura de los datos para un gráfico de Chart.js."""
    type: str
    labels: list[str]
    datasets: list[Dict[str, Any]]

class RespuestaRAG(BaseModel):
    """Define la nueva estructura de la respuesta de la API."""
    texto: str
    grafico_data: Optional[GraficoData] = None
    
# Define la estructura de la solicitud
class Consulta(BaseModel):
    pregunta: str

def parsear_respuesta_llm(respuesta_llm: str) -> RespuestaRAG:
    """
    Busca el separador en la respuesta del LLM para separar el texto del JSON del gráfico.
    """
    separador = "[GRAFICO_JSON]"
    if separador in respuesta_llm:
        partes = respuesta_llm.split(separador, 1)
        texto = partes[0].strip()
        json_str = partes[1].strip()
        
        try:
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json\n", "").replace("\n```", "")

            grafico_data = json.loads(json_str)
            return RespuestaRAG(texto=texto, grafico_data=grafico_data)
        except json.JSONDecodeError:
            # Si el JSON es inválido, devolvemos solo el texto completo
            return RespuestaRAG(texto=respuesta_llm)
    else:
        # No hay gráfico, devolvemos solo el texto
        return RespuestaRAG(texto=respuesta_llm.strip())

app = FastAPI(
    title="API RAG Agrocadenas con Gemini",
    description="Backend para consultar datos de trazabilidad y optimización."
)


origins = [
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/consulta/", response_model=RespuestaRAG)
async def consulta_rag_api(consulta: Consulta):
    """
    Endpoint que recibe una pregunta, la procesa con el LLM y devuelve
    una respuesta estructurada con texto y, opcionalmente, datos para un gráfico.
    """
    try:
        respuesta_cruda = query_engine.query(consulta.pregunta)
        respuesta_estructurada = parsear_respuesta_llm(str(respuesta_cruda))
        return respuesta_estructurada
    except Exception as e:
        return RespuestaRAG(texto=f"Error al procesar la consulta: {e}")
    
# --- EJECUCIÓN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

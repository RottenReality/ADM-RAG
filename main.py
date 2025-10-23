import json
import os
import re
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings # <-- Agregado 'Settings'
from llama_index.core.schema import TextNode
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.mistralai import MistralAI
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

def obtener_llm(llm_name: str):
    """Devuelve la instancia del LLM según el nombre recibido."""
    llm_name_lower = llm_name.lower()

    if "gemini" in llm_name_lower:
        return Gemini(model=llm_name, api_key=os.getenv("GEMINI_API_KEY"))
    elif "gpt" in llm_name_lower or "openai" in llm_name_lower:
        return Groq(model=llm_name, api_key=os.getenv("GROQ_API_KEY"))
    elif "llama" in llm_name_lower:
        return Groq(model=llm_name, api_key=os.getenv("GROQ_API_KEY"))
    elif "moonshotai" in llm_name_lower or "kimi" in llm_name_lower:
        return Groq(model=llm_name, api_key=os.getenv("GROQ_API_KEY"))
    else:
        raise ValueError(f"Modelo LLM no soportado: {llm_name}")

llm = obtener_llm("gemini-2.5-flash")
Settings.llm = llm

# Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 
Settings.embed_model = embed_model

# ChromaDB
print("Inicializando ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Crear el Índice RAG
print("Creando índice vectorial...")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex(
    nodes=nodos_rag, 
    embed_model=embed_model, 
    vector_store=vector_store
)
print("¡Indexación completa con LLM y ChromaDB!")

qa_prompt_template_str = (
    "Eres un asistente experto en análisis de datos de agrocadenas.\n"
    "Tu tarea es analizar información sobre procesos productivos y responder preguntas con precisión y estructura clara.\n\n"
    "Contexto de la cadena de suministro:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Pregunta del usuario: {query_str}\n\n"
    "--- INSTRUCCIONES DE FORMATO ---\n"
    "1. Comienza SIEMPRE con una respuesta textual clara, concisa y en lenguaje natural. No uses listas numeradas ni viñetas innecesarias si no aportan al análisis.\n"
    "2. Si la pregunta involucra valores numéricos, comparaciones, distribuciones o cualquier información que pueda representarse visualmente "
    "(por ejemplo: tiempos, porcentajes, cantidades, etapas, etc.), DEBES incluir un bloque de gráfico.\n"
    "3. Para incluir el gráfico, escribe un salto de línea y luego el separador exacto:\n"
    "   [GRAFICO_JSON]\n"
    "   En la línea siguiente, coloca un bloque de código JSON **válido y bien formado**, sin texto adicional ni comentarios.\n"
    "4. El JSON debe seguir este formato y ser compatible con Chart.js:\n"
    "   {\n"
    "       'type': 'bar'|'pie'|'line',\n"
    "       'labels': ['etiqueta1', 'etiqueta2', ...],\n"
    "       'datasets': [\n"
    "           { 'label': 'Nombre del conjunto de datos', 'data': [valor1, valor2, ...] }\n"
    "       ]\n"
    "   }\n"
    "5. Si la respuesta es puramente textual (no hay datos cuantitativos o comparativos), **NO** incluyas el separador ni el JSON.\n"
    "6. No incluyas tablas Markdown, HTML ni código adicional fuera del JSON del gráfico.\n"
    "7. Asegúrate de que los valores en el JSON coincidan con los números mencionados en tu respuesta textual.\n"
    "8. En el bloque JSON usa siempre comillas dobles (\") válidas para JSON estándar."
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
    
class Consulta(BaseModel):
    pregunta: str
    llm_name: str = "gemini-2.5-flash"

def parsear_respuesta_llm(respuesta_llm: str) -> RespuestaRAG:
    """
    Busca el separador en la respuesta del LLM para separar el texto del JSON del gráfico.
    Incluye validaciones robustas y tolerancia a formatos mal formados (especialmente en Llama/Mistral).
    """
    separador = "[GRAFICO_JSON]"
    if separador not in respuesta_llm:
        return RespuestaRAG(texto=respuesta_llm.strip())

    partes = respuesta_llm.split(separador, 1)
    texto = partes[0].strip()
    json_str = partes[1].strip()

    json_str = json_str.replace("```json", "").replace("```", "").strip()
    json_str = json_str.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    if "'" in json_str and '"' not in json_str:
        json_str = json_str.replace("'", '"')

    match = re.search(r'(\{.*\}|\[.*\])', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)

    try:
        grafico_data = json.loads(json_str)

        if not isinstance(grafico_data, dict) or \
           "labels" not in grafico_data or \
           "datasets" not in grafico_data:
            return RespuestaRAG(texto=texto + "\n\n⚠️ El gráfico no se pudo interpretar correctamente.")

        return RespuestaRAG(texto=texto, grafico_data=grafico_data)

    except json.JSONDecodeError as e:
        json_str_reparado = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            grafico_data = json.loads(json_str_reparado)
            if "labels" in grafico_data and "datasets" in grafico_data:
                return RespuestaRAG(texto=texto, grafico_data=grafico_data)
        except Exception:
            pass
        return RespuestaRAG(texto=texto + "\n\n⚠️ El gráfico no se pudo interpretar correctamente.")

    except Exception as e:
        return RespuestaRAG(texto=f"{texto}\n\n⚠️ Error al interpretar gráfico: {e}")



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
        if consulta.llm_name:
            Settings.llm = obtener_llm(consulta.llm_name)

        print(f"🧠 Usando modelo: {Settings.llm.__class__.__name__} ({consulta.llm_name})")

        respuesta_cruda = query_engine.query(consulta.pregunta)
        respuesta_estructurada = parsear_respuesta_llm(str(respuesta_cruda))
        return respuesta_estructurada
    except Exception as e:
        return RespuestaRAG(texto=f"Error al procesar la consulta: {e}")

@app.post("/comparar/", response_model=List[RespuestaRAG])
async def comparar_llms(consulta: Consulta):
    """
    Ejecuta la misma pregunta en los 4 modelos y devuelve todas las respuestas.
    """
    modelos = [
        "gemini-2.5-flash",
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "moonshotai/kimi-k2-instruct"
    ]
    respuestas = []

    for modelo in modelos:
        try:
            llm_actual = obtener_llm(modelo)
            query_engine_temp = index.as_query_engine(
                llm=llm_actual,
                similarity_top_k=10,
                text_qa_template=qa_prompt_template
            )

            print(f"🤖 Consultando con {modelo}...")
            respuesta_cruda = query_engine_temp.query(consulta.pregunta)

            respuesta_estructurada = parsear_respuesta_llm(str(respuesta_cruda))
            respuesta_estructurada.texto = f"[{modelo}] {respuesta_estructurada.texto}"
            respuestas.append(respuesta_estructurada)

        except Exception as e:
            respuestas.append(
                RespuestaRAG(texto=f"[{modelo}] ⚠️ Error: {e}")
            )

    return respuestas

# --- EJECUCIÓN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

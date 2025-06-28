# main.py
import os
from dotenv import load_dotenv
from datetime import date
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# --- Importaciones de LangChain ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain.tools import StructuredTool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuración Inicial ---

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: La variable de entorno OPENAI_API_KEY no está configurada.")

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

app = FastAPI(title="API del Chatbot del Camping")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- Conexión a la Base de Datos y Herramientas ---
db = SQLDatabase.from_uri("sqlite:///rosaleda_camping.db")


# --- Herramientas de Creación y Modificación ---
class ReservationInput(BaseModel):
    nombre_alojamiento: str = Field(description="El nombre exacto del tipo de alojamiento. Ej: 'Bungalow Luxe', 'Parcela Grande'.")
    fecha_inicio: str = Field(description="Fecha de inicio en formato<y_bin_564>-MM-DD.")
    fecha_fin: str = Field(description="Fecha de fin en formato<y_bin_564>-MM-DD.")
    nombre_cliente: str = Field(description="Nombre completo del cliente.")

def crear_reserva(nombre_alojamiento: str, fecha_inicio: str, fecha_fin: str, nombre_cliente: str) -> str:
    """Crea una nueva reserva en la base de datos."""
    try:
        id_alojamiento_result = db.run(f"SELECT id FROM alojamientos WHERE nombre = '{nombre_alojamiento}' LIMIT 1")
        if not id_alojamiento_result or not id_alojamiento_result.strip('[]() ,'):
             return f"Error: No se encontró un tipo de alojamiento llamado '{nombre_alojamiento}'."
        id_alojamiento = int(id_alojamiento_result.strip('[]() ,'))
        db.run(f"INSERT INTO reservas (id_alojamiento, fecha_inicio, fecha_fin, nombre_cliente) VALUES ({id_alojamiento}, '{fecha_inicio}', '{fecha_fin}', '{nombre_cliente}')")
        return f"¡Reserva creada con éxito para {nombre_cliente}!"
    except Exception as e:
        return f"Error al crear la reserva: {e}."

make_reservation_tool = StructuredTool.from_function(func=crear_reserva, name="CrearReserva", description="Útil para crear una nueva reserva.", args_schema=ReservationInput)


# --- Herramientas de Consulta ---
sql_agent_executor = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)
sql_tool = Tool(name="ConsultarBaseDeDatos", func=sql_agent_executor.invoke, description="Imprescindible para cualquier pregunta sobre precios, disponibilidad de fechas, reservas existentes, clientes y detalles de capacidad de los alojamientos. Debe usarse para consultar las tablas 'alojamientos', 'tarifas' y 'reservas'.")

# --- CAMBIO: Texto del documento de conocimiento ampliado con toda la información del PDF ---
conocimiento_general_texto = """
Información General del Camping La Rosaleda:

**Normativa General Aplicable a Todos:**
- Silencio obligatorio de 00:00 a 08:00.
- Es obligatorio llevar siempre puesta la pulsera identificativa.
- Está prohibido superar el aforo máximo de cada alojamiento, contando a los bebés.
- Solo se permite un vehículo por alojamiento.
- Prohibido colgar hamacas o cuerdas en árboles o farolas.

**Detalles del Bungalow Luxe:**
- **Capacidad:** Máximo 4 adultos y 1 niño.
- **Animales:** No se permiten animales.
- **Check-in/out:** Entrada desde 16:00. Salida antes de 11:00 (20/06-07/09) o antes de 12:00 (resto del año).
- **Incluye:** Cocina equipada, aire acondicionado, TV, terraza, salón, baño, jardín y ropa de cama. No incluye toallas.
- **Electricidad:** 15 kW/día incluidos. Exceso se cobra a 0.55 €/kWh. En estancias de más de 14 noches en temporada baja, se paga todo el consumo.
- **Fianza y Suplementos:** Fianza de 70€ (bloqueo en tarjeta). Pérdida de llave: 20€. Limpieza por fumar: 30€.
- **Prohibiciones:** Prohibido hacer barbacoas. Se permite plancha eléctrica en el porche. Prohibido poner objetos en el césped.

**Detalles de la Parcela Grande:**
- **Capacidad:** Hasta 6 personas (niños y bebés incluidos).
- **Animales:** Mascotas permitidas solo del 01/09 al 30/06, con carné de vacunación y chip. Prohibidas del 01/07 al 31/08.
- **Check-in/out:** Entrada desde 13:00. Salida antes de 12:00.
- **Electricidad:** 5 kW/día incluidos (10 kW/día del 20/06 al 07/09). Exceso a 0.55 €/kWh.
- **Fianza:** 20€ (preautorización en tarjeta).
- **Suplementos:** Persona adicional desde 7€/noche.
- **Normas:** Prohibido cambiarse de parcela sin autorización. Se puede cocinar con gas o plancha eléctrica (no fuego directo en temporada alta).
"""
docs = [Document(page_content=conocimiento_general_texto)]
retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()
retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("Responde a la pregunta basándote únicamente en el siguiente contexto:\n\n<contexto>{context}</contexto>\n\nPregunta: {input}")))
rag_tool = Tool(
    name="ConsultarInformacionGeneral",
    func=lambda query: retrieval_chain.invoke({"input": query})['answer'],
    description="Útil para responder preguntas sobre normativas (silencio, pulseras, mascotas), fianzas, suplementos (llave, limpieza), horarios (check-in/out), qué incluye un alojamiento, y otras políticas generales que no están en la base de datos."
)


# --- Creación del Agente Principal ---
tools = [sql_tool, rag_tool, make_reservation_tool]
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente virtual muy útil para la recepción del camping 'La Rosaleda'. Responde siempre en español. La fecha de hoy es {today}. Utiliza el historial de chat para entender el contexto. Para obtener precios, debes consultar la tabla 'tarifas' y unirla con 'alojamientos'. Para normativas y suplementos, usa la herramienta de información general."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Rutas de la API (sin cambios) ---
class UserInput(BaseModel):
    pregunta: str
    historial: List[Dict[str, Any]] = []

@app.post("/chat")
async def chat_with_agent(user_input: UserInput):
    today_date = date.today().strftime("%Y-%m-%d")
    chat_history = [HumanMessage(content=msg['text']) if msg['sender'] == 'user' else AIMessage(content=msg['text']) for msg in user_input.historial]
    response = await agent_executor.ainvoke({"input": user_input.pregunta, "today": today_date, "chat_history": chat_history})
    return {"respuesta": response["output"]}

@app.get("/")
def read_root(): return {"mensaje": "Bienvenido a la API del Chatbot del Camping."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

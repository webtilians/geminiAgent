# main.py
import os
from dotenv import load_dotenv
from datetime import date
from typing import List, Dict, Any
from ast import literal_eval # Importar para parsear la salida de la DB

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# --- Importaciones de LangChain ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain.tools import StructuredTool
from langchain_community.utilities import SQLDatabase
# Se elimina create_sql_agent porque se reemplaza con herramientas específicas
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
    fecha_inicio: str = Field(description="Fecha de inicio en formato ইয়াহিয়া-MM-DD.")
    fecha_fin: str = Field(description="Fecha de fin en formato ইয়াহিয়া-MM-DD.")
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

class ModifyReservationInput(BaseModel):
    nombre_cliente: str = Field(description="Nombre del cliente cuya reserva se quiere modificar.")
    nueva_fecha_inicio: str = Field(description="La nueva fecha de inicio para la reserva en formato ইয়াহিয়া-MM-DD.")
    nueva_fecha_fin: str = Field(description="La nueva fecha de fin para la reserva en formato ইয়াহিয়া-MM-DD.")

def modificar_reserva(nombre_cliente: str, nueva_fecha_inicio: str, nueva_fecha_fin: str) -> str:
    """Busca una reserva por nombre de cliente y actualiza sus fechas."""
    try:
        reserva_existente = db.run(f"SELECT id FROM reservas WHERE nombre_cliente LIKE '%{nombre_cliente}%' LIMIT 1")
        if not reserva_existente or not reserva_existente.strip('[]() ,'):
            return f"No se encontró ninguna reserva a nombre de {nombre_cliente}. Por favor, verifica el nombre."
        id_reserva = int(reserva_existente.strip('[]() ,'))
        db.run(f"UPDATE reservas SET fecha_inicio = '{nueva_fecha_inicio}', fecha_fin = '{nueva_fecha_fin}' WHERE id = {id_reserva}")
        return f"¡Reserva de {nombre_cliente} actualizada con éxito! Nuevas fechas: del {nueva_fecha_inicio} al {nueva_fecha_fin}."
    except Exception as e:
        return f"Error al modificar la reserva: {e}."

modify_reservation_tool = StructuredTool.from_function(func=modificar_reserva, name="ModificarReserva", description="Útil para modificar las fechas de una reserva existente.", args_schema=ModifyReservationInput)


# --- HERRAMIENTA DEFINITIVA: Listar Todas las Reservas ---
def listar_todas_las_reservas(query: str = "") -> str:
    """Recupera y formatea TODAS las reservas de la base de datos."""
    try:
        query_sql = "SELECT r.id, a.nombre, r.fecha_inicio, r.fecha_fin, r.nombre_cliente FROM reservas r JOIN alojamientos a ON r.id_alojamiento = a.id ORDER BY r.id;"
        resultados_str = db.run(query_sql)
        if not resultados_str or resultados_str == '[]': return "No hay ninguna reserva en la base de datos."
        lista_reservas = literal_eval(resultados_str)
        if not lista_reservas: return "No hay ninguna reserva en la base de datos."
        respuesta_formateada = "Aquí tienes la lista completa de todas las reservas:\n\n"
        for reserva in lista_reservas:
            reserva_id, alojamiento_nombre, fecha_inicio, fecha_fin, cliente_nombre = reserva
            respuesta_formateada += f"**Reserva ID {reserva_id}**\n- Cliente: {cliente_nombre}\n- Alojamiento: {alojamiento_nombre}\n- Fechas: del {fecha_inicio} al {fecha_fin}\n\n"
        return respuesta_formateada.strip()
    except Exception as e:
        return f"Se ha producido un error al intentar recuperar las reservas: {e}"

list_reservations_tool = Tool.from_function(func=listar_todas_las_reservas, name="ListarTodasLasReservas", description="La única herramienta a usar cuando el usuario pide ver 'todas las reservas', 'la lista de reservas' o una consulta similar para obtener un listado completo.")


# --- Herramienta de Consulta de Información General ---
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
rag_tool = Tool(name="ConsultarInformacionGeneral", func=lambda query: retrieval_chain.invoke({"input": query})['answer'], description="Útil para responder preguntas sobre normativas, fianzas, suplementos, horarios, y políticas generales.")


# --- Creación del Agente Principal ---
tools = [list_reservations_tool, rag_tool, make_reservation_tool, modify_reservation_tool]
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un asistente virtual para la recepción del camping 'La Rosaleda'. Responde siempre en español.
    La fecha de hoy es {today}.
    INSTRUCCIONES DE HERRAMIENTAS:
    - Para listar TODAS las reservas, usa SIEMPRE la herramienta `ListarTodasLasReservas`.
    - Para crear una reserva, usa `CrearReserva`.
    - Para modificar una reserva, usa `ModificarReserva`.
    - Para cualquier otra pregunta sobre normativas, fianzas, suplementos, o políticas, usa `ConsultarInformacionGeneral`.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# --- Rutas de la API ---
class UserInput(BaseModel):
    pregunta: str
    historial: List[Dict[str, Any]] = []

# --- CAMBIO: La ruta /chat ahora devuelve los pasos intermedios ---
@app.post("/chat")
async def chat_with_agent(user_input: UserInput):
    today_date = date.today().strftime("%Y-%m-%d")
    chat_history = [HumanMessage(content=msg['text']) if msg['sender'] == 'user' else AIMessage(content=msg['text']) for msg in user_input.historial]
    
    response = await agent_executor.ainvoke({
        "input": user_input.pregunta,
        "today": today_date,
        "chat_history": chat_history
    })
    
    # Extraer los pasos intermedios para enviarlos al frontend
    intermediate_steps = response.get("intermediate_steps", [])
    formatted_steps = []
    for action, observation in intermediate_steps:
        tool_name = getattr(action, 'tool', 'Unknown Tool')
        formatted_steps.append(f"Usando herramienta: **{tool_name}**")

    return {
        "respuesta": response.get("output", "No se obtuvo respuesta."),
        "pasos_intermedios": formatted_steps
    }

@app.get("/")
def read_root(): return {"mensaje": "Bienvenido a la API del Chatbot del Camping."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# main.py
# Instala las dependencias necesarias con:
# pip install fastapi uvicorn python-dotenv openai langchain langchain-openai langchain-community faiss-cpu sqlalchemy

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


# --- Base de datos y herramientas ---

db = SQLDatabase.from_uri("sqlite:///camping.db")

def setup_database(database: SQLDatabase):
    """Crea tablas y añade datos de ejemplo si no existen."""
    parcelas_schema = "CREATE TABLE IF NOT EXISTS parcelas (id INTEGER PRIMARY KEY, tipo TEXT NOT NULL, precio_noche REAL NOT NULL, electricidad BOOLEAN NOT NULL)"
    reservas_schema = "CREATE TABLE IF NOT EXISTS reservas (id INTEGER PRIMARY KEY, id_parcela INTEGER, fecha_inicio TEXT NOT NULL, fecha_fin TEXT NOT NULL, nombre_cliente TEXT, FOREIGN KEY(id_parcela) REFERENCES parcelas(id))"
    database.run(parcelas_schema)
    database.run(reservas_schema)
    if not database.run("SELECT * FROM parcelas;"):
        database.run("INSERT INTO parcelas (tipo, precio_noche, electricidad) VALUES ('Tienda Pequeña', 15.50, 0), ('Tienda Grande', 25.00, 1), ('Caravana', 35.75, 1), ('Bungalow', 75.00, 1);")
    if not database.run("SELECT * FROM reservas;"):
         database.run("INSERT INTO reservas (id_parcela, fecha_inicio, fecha_fin, nombre_cliente) VALUES (1, '2025-06-20', '2025-06-30', 'Maria Garcia'), (2, '2025-07-20', '2025-07-25', 'Juan Pérez'), (4, '2025-08-05', '2025-08-15', 'Ana López');")

setup_database(db)
print("Base de datos de ejemplo configurada.")


# --- Herramientas de Creación y Modificación ---
class ReservationInput(BaseModel):
    tipo_parcela: str = Field(description="El tipo de alojamiento que el cliente desea. Debe ser uno de: 'Tienda Pequeña', 'Tienda Grande', 'Caravana', 'Bungalow'.")
    fecha_inicio: str = Field(description="Fecha de inicio de la reserva en formato YYYY-MM-DD.")
    fecha_fin: str = Field(description="Fecha de fin de la reserva en formato YYYY-MM-DD.")
    nombre_cliente: str = Field(description="Nombre completo de la persona que realiza la reserva.")

def crear_reserva(tipo_parcela: str, fecha_inicio: str, fecha_fin: str, nombre_cliente: str) -> str:
    """Función para crear una nueva reserva en la base de datos."""
    try:
        # --- CAMBIO: Búsqueda más robusta del tipo de parcela ---
        clean_tipo = tipo_parcela.lower()
        search_keyword = ""
        if "bungalow" in clean_tipo:
            search_keyword = "Bungalow"
        elif "caravana" in clean_tipo:
            search_keyword = "Caravana"
        elif "tienda grande" in clean_tipo:
            search_keyword = "Tienda Grande"
        elif "tienda pequeña" in clean_tipo:
            search_keyword = "Tienda Pequeña"
        else:
            # Si no encontramos una palabra clave, puede que el agente ya haya enviado el tipo correcto.
            search_keyword = tipo_parcela

        id_parcela_result = db.run(f"SELECT id FROM parcelas WHERE tipo = '{search_keyword}' LIMIT 1")
        
        # --- CAMBIO CLAVE: Comprobar si la consulta devolvió algo ANTES de intentar la conversión ---
        if not id_parcela_result or not id_parcela_result.strip('[]() ,'):
             return f"Error: No se encontró un tipo de parcela que coincida con '{tipo_parcela}'. Los tipos válidos son: 'Tienda Pequeña', 'Tienda Grande', 'Caravana', 'Bungalow'."
        
        # Si llegamos aquí, es seguro convertir a entero.
        id_parcela = int(id_parcela_result.strip('[]() ,'))
        
        db.run(f"INSERT INTO reservas (id_parcela, fecha_inicio, fecha_fin, nombre_cliente) VALUES ({id_parcela}, '{fecha_inicio}', '{fecha_fin}', '{nombre_cliente}')")
        return f"¡Reserva creada con éxito a nombre de {nombre_cliente} para una {search_keyword} desde el {fecha_inicio} hasta el {fecha_fin}!"
    except Exception as e:
        return f"Error al crear la reserva: {e}. Por favor, verifica que todos los datos son correctos."

make_reservation_tool = StructuredTool.from_function(func=crear_reserva, name="CrearReserva", description="Útil para crear una nueva reserva.", args_schema=ReservationInput)

class ModifyReservationInput(BaseModel):
    nombre_cliente: str = Field(...)
    nueva_fecha_inicio: str = Field(...)
    nueva_fecha_fin: str = Field(...)

def modificar_reserva(nombre_cliente: str, nueva_fecha_inicio: str, nueva_fecha_fin: str) -> str:
    try:
        reserva_existente = db.run(f"SELECT id FROM reservas WHERE nombre_cliente LIKE '%{nombre_cliente}%' LIMIT 1")
        if not reserva_existente or not reserva_existente.strip('[]() ,'):
            return f"No se encontró ninguna reserva a nombre de {nombre_cliente}."
        id_reserva = int(reserva_existente.strip('[]() ,'))
        db.run(f"UPDATE reservas SET fecha_inicio = '{nueva_fecha_inicio}', fecha_fin = '{nueva_fecha_fin}' WHERE id = {id_reserva}")
        return f"¡Reserva de {nombre_cliente} actualizada!"
    except Exception as e:
        return f"Error: {e}."
modify_reservation_tool = StructuredTool.from_function(func=modificar_reserva, name="ModificarReserva", description="Útil para modificar las fechas de una reserva existente.", args_schema=ModifyReservationInput)

# Herramientas de Consulta (sin cambios)
sql_agent_executor = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)
sql_tool = Tool(name="ConsultarBaseDeDatos", func=sql_agent_executor.invoke, description="Imprescindible para consultar precios, disponibilidad, reservas existentes, etc.")
docs = [Document(page_content="Normas de la piscina..."), Document(page_content="Actividades y excursiones...")]
retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()
retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("Contexto: {context}\n\nPregunta: {input}")))
rag_tool = Tool(name="ConsultarInformacionGeneral", func=lambda query: retrieval_chain.invoke({"input": query})['answer'], description="Útil para consultar normas, reglas y actividades.")

# --- Creación del Agente Principal ---
tools = [sql_tool, rag_tool, make_reservation_tool, modify_reservation_tool]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente virtual muy útil para la recepción de un camping. Responde siempre en español. La fecha de hoy es {today}. Utiliza el historial de chat para entender el contexto."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Rutas de la API ---
class UserInput(BaseModel):
    pregunta: str
    historial: List[Dict[str, Any]] = []

@app.post("/chat")
async def chat_with_agent(user_input: UserInput):
    today_date = date.today().strftime("%Y-%m-%d")

    chat_history = []
    for msg in user_input.historial:
        if msg.get('sender') == 'user':
            chat_history.append(HumanMessage(content=msg.get('text', '')))
        elif msg.get('sender') == 'bot':
            chat_history.append(AIMessage(content=msg.get('text', '')))

    response = await agent_executor.ainvoke({
        "input": user_input.pregunta,
        "today": today_date,
        "chat_history": chat_history
    })
    return {"respuesta": response["output"]}

@app.get("/")
def read_root(): return {"mensaje": "Bienvenido a la API del Chatbot del Camping."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

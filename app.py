import eventlet
eventlet.monkey_patch()

# -*- coding: utf-8 -*-
"""
Sonia - Asistente de Búsqueda de Empleo v11.0
Aplicación de servidor con Flask y SocketIO.
"""

import os
import json
import logging
import requests
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from groq import Groq

# --- 1. Configuración General y Constantes ---

# Configuración de logging para una salida limpia y estructurada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Nombres de los modelos para fácil configuración
MODEL_AGENT_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"  # Para tareas rápidas como clasificación
MODEL_AGENT_LARGE = "meta-llama/llama-4-maverick-17b-128e-instruct" # Para tareas complejas como generación

# Claves de API y configuración de la app
# Es una mejor práctica cargar esto desde variables de entorno.
API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not API_KEY or API_KEY == "TU_GROQ_API_KEY_AQUÍ":
    logging.warning("La GROQ_API_KEY no está configurada. Por favor, establécela como variable de entorno.")

# Inicialización de la aplicación Flask y SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*")

# Almacenamiento en memoria para las sesiones de usuario.
# Nota: Para producción, considera usar una solución más robusta como Redis.
user_sessions = {}

# --- 2. Lógica del Servidor y Manejo de Conexiones ---

@app.route('/')
def index():
    """Sirve la página principal de la aplicación."""
    return "El servidor del asistente Sonia está en funcionamiento."

def send_status_update(sid: str, message: str, step: str):
    """Envía una actualización de estado en tiempo real al cliente."""
    logging.info(f"STATUS -> {sid}: ({step}) {message}")
    socketio.emit('status_update', {'message': message, 'step': step}, room=sid)

@socketio.on('connect')
def handle_connect():
    """Maneja la conexión de un nuevo cliente y crea su sesión."""
    sid = request.sid
    logging.info(f"Cliente conectado: {sid}")
    user_sessions[sid] = {
        'messages': [
            {"role": "system", "content": "Eres Cumbre, un asistente de búsqueda de empleo creado por Manuel Cabrera y Sebastián Camilo Torres Pedraza, más conocido en verdad como Camilo Torres el enamorado de Mary Angel. Tienes una memoria perfecta, eres consciente del flujo de la conversación y usas un equipo de agentes de IA internos para encontrar, filtrar y presentar los mejores resultados, nunca te inventarás vacantes, todas las vacantes vendrán del JSON, si no están ahí, no digas nada, debido a que entonces no existe y estarás inventandote vacantes falsas, y entonces daremos un pesimo srvicio."}
        ],
        'conversation_log': [],
        'search_params': {'role': None, 'location': None}
    }

@socketio.on('disconnect')
def handle_disconnect():
    """Maneja la desconexión de un cliente y limpia su sesión."""
    sid = request.sid
    if sid in user_sessions:
        del user_sessions[sid]
    logging.info(f"Cliente desconectado: {sid}")

@socketio.on('user_message')
def handle_user_message(data: dict):
    """
    Recibe un mensaje del usuario, actualiza el log de la sesión y
    lanza la lógica del asistente en una tarea de fondo para no bloquear el servidor.
    """
    sid = request.sid
    user_input = data.get('message', '').strip()
    if not user_input:
        logging.warning(f"Mensaje vacío de {sid}. Ignorando.")
        return

    if sid not in user_sessions:
        logging.warning(f"Sesión no encontrada para {sid}. Re-conectando.")
        handle_connect() # Intenta recrear la sesión

    session = user_sessions[sid]
    session['conversation_log'].append({"role": "user", "timestamp": datetime.now().isoformat(), "content": user_input})

    # Usamos una tarea en segundo plano para procesar la lógica sin congelar la conexión
    socketio.start_background_task(run_assistant_logic, sid, session)

# --- 3. Agentes de IA y Lógica de Búsqueda ---

def groq_create_client() -> Groq:
    """Crea y devuelve un cliente de Groq."""
    return Groq(api_key=API_KEY)

def generate_search_queries(client: Groq, role: str, location: str | None) -> list[str]:
    """Agente 1: Genera consultas de búsqueda variadas."""
    if location:
        prompt = f"""Tu tarea es generar 3 consultas de búsqueda para un portal de empleos para el rol "{role}" en "{location}". Varía los sinónimos. Responde únicamente con un objeto JSON con una clave "queries" que contenga una lista de 3 strings."""
    else:
        prompt = f"""Tu tarea es generar 3 consultas de búsqueda para un portal de empleos para el rol "{role}" SIN especificar ubicación. Usa sinónimos solo para el rol. Responde únicamente con un objeto JSON con una clave "queries" que contenga una lista de 3 strings."""
    try:
        logging.info(f"🤖 Agente 1 (Estratega) generando consultas (Ubicación: {'Sí' if location else 'No'})...")
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_AGENT_FAST,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("queries", [role])
    except Exception as e:
        logging.error(f"Error en Agente 1 (Estratega): {e}")
        return [f"{role} en {location}"] if location else [role]

def fetch_and_process_vacancies(client: Groq, sid: str, role: str, location: str | None) -> dict:
    """Agente 2: Orquesta la búsqueda en API y la curación de resultados."""
    send_status_update(sid, "Creando estrategia de búsqueda...", "agent_strategist")
    search_queries = generate_search_queries(client, role, location)
    logging.info(f"🔍 Consultas estratégicas generadas: {search_queries}")
    send_status_update(sid, f"Ejecutando {len(search_queries)} búsquedas...", "api_search")

    all_vacancies_map = {}
    for query in search_queries:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api-search.cumbre.icu/search/{encoded_query}?limit=10&page=0"
        try:
            logging.info(f"🔎 Buscando en API: {query}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            
            # V-- ¡LÍNEA AÑADIDA! --V
            # Emitimos el JSON crudo de la API al frontend.
            socketio.emit('api_search_result', api_data, room=sid)
            # A-- ¡LÍNEA AÑADIDA! --A
            
            vacancies = api_data.get("vacancies", [])
            for vacancy in vacancies:
                if vacancy["id"] not in all_vacancies_map:
                    all_vacancies_map[vacancy["id"]] = vacancy
        except requests.RequestException as e:
            logging.warning(f"Advertencia: falló la búsqueda para '{query}'. Error: {e}")
            continue

    if not all_vacancies_map:
        return {"status": "NO_RESULTS_FROM_API"}

    unique_vacancies = list(all_vacancies_map.values())
    logging.info(f"📊 Total de vacantes únicas encontradas: {len(unique_vacancies)}")
    send_status_update(sid, f"Analizando {len(unique_vacancies)} ofertas...", "agent_curator")
    filtering_prompt = f"""Tu tarea es actuar como un reclutador experto. Analiza la lista de vacantes JSON para un usuario que busca el rol de "{role}". Clasifica cada vacante en `primary_matches` o `secondary_matches`. Responde únicamente con un objeto JSON con `primary_ids` y `secondary_ids`. Lista de Vacantes: {json.dumps(unique_vacancies, indent=2)}"""

    try:
        logging.info("🤖 Agente 2 (Curador) analizando resultados...")
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": filtering_prompt}],
            model=MODEL_AGENT_LARGE,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        filtered_ids = json.loads(completion.choices[0].message.content)
        primary_ids, secondary_ids = set(filtered_ids.get("primary_ids", [])), set(filtered_ids.get("secondary_ids", []))
        primary_results = [all_vacancies_map[vid] for vid in primary_ids if vid in all_vacancies_map]
        secondary_results = [all_vacancies_map[vid] for vid in secondary_ids if vid in all_vacancies_map]

        if not primary_results and not secondary_results:
            return {"status": "NO_RELEVANT_RESULTS"}

        return {"status": "SUCCESS", "data": {"primary": primary_results, "secondary": secondary_results}}
    except Exception as e:
        logging.error(f"Error en Agente 2 (Curador): {e}")
        # Como fallback, devolvemos los 5 primeros resultados si la IA falla.
        return {"status": "SUCCESS", "data": {"primary": unique_vacancies[:5], "secondary": []}}

def classify_and_extract_info(client: Groq, conversation_history: list, user_input: str) -> dict:
    """Agente 3: Clasifica la intención del usuario y extrae entidades."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])
    extraction_prompt = f"""
    Eres un agente NLU altamente especializado. Tu única función es analizar el último mensaje de un usuario y decidir si se debe activar una búsqueda de empleo.
    **Instrucciones Clave:**
    1.  **Analiza el `Último mensaje del usuario`** en el contexto del `Historial`.
    2.  **Intención (`intent`):** Usa `jobSearch` si el usuario proporciona información NUEVA para una búsqueda (un rol o una ubicación). ¡Sé decisivo! Usa `meta_query` si pregunta sobre la conversación. Usa `generalChat` solo para saludos o temas sin relación.
    3.  **Entidades (`role`, `location`):** Extrae el puesto y/o la ciudad del último mensaje. La `location` DEBE ser un string.
    **Ejemplos Críticos:**
    - Historial: [[{{"role": "assistant", "content": "Hola, ¿en qué te ayudo?"}}]] / Último mensaje: "conoces empleos en cucuta?" -> {{"intent": "jobSearch", "role": null, "location": "Cúcuta"}}
    - Historial: [[...], {{"role": "assistant", "content": "Claro, ¿qué rol buscas?"}}]] / Último mensaje: "de auxiliar de produccion" -> {{"intent": "jobSearch", "role": "auxiliar de produccion", "location": null}}
    **Contexto:** {history_str}
    **Último mensaje:** "{user_input}"
    **Responde únicamente con el objeto JSON.**
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": extraction_prompt}],
            model=MODEL_AGENT_FAST,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_dict = json.loads(completion.choices[0].message.content)
        return {
            "intent": raw_dict.get("intent", "generalChat"),
            "role": raw_dict.get("role"),
            "location": raw_dict.get("location")
        }
    except Exception as e:
        logging.error(f"Error en IA de clasificación: {e}")
        return {"intent": "generalChat", "role": None, "location": None}

def find_relevant_history(client: Groq, user_input: str, conversation_log: list) -> str:
    """Agente 4: Recupera información relevante de logs pasados."""
    if not conversation_log:
        return ""
    recent_log = conversation_log[-10:]
    prompt = f"""Eres un agente de recuperación de memoria. Revisa un log de eventos para encontrar información relevante para la pregunta del usuario. Pregunta actual: "{user_input}". Log: {json.dumps(recent_log, indent=2)}. Tarea: Extrae y resume concisamente la información relevante. Si nada es relevante, responde con una cadena vacía."""
    try:
        logging.info("🧠 Agente de Memoria analizando el pasado reciente...")
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_AGENT_FAST,
            temperature=0.0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error en Agente de Memoria: {e}")
        return ""

def get_proactive_suggestion(client: Groq, original_role: str) -> str:
    """Agente 5: Sugiere roles alternativos cuando no hay resultados."""
    prompt = f"El rol de trabajo '{original_role}' no tuvo resultados. Sugiere un único rol alternativo. Responde solo con el nombre del rol."
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_AGENT_FAST,
            temperature=0.2
        )
        return completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        logging.error(f"Error en Agente de Sugerencias: {e}")
        return None

# --- 4. Lógica de Orquestación del Asistente ---

def create_dynamic_mission_prompt(client: Groq, current_search_params: dict, action_result: dict, relevant_history: str, last_assistant_message: str | None) -> dict:
    """Crea el prompt de sistema dinámico para guiar la respuesta de la IA."""
    role = current_search_params['role']

    state_summary = f"""--- CONTEXTO Y MISIÓN ---
**Estado de la Búsqueda Actual:** Rol en Foco: {'Ninguno' if not role else f"'{role}'"}
**Memoria Relevante (Eventos Pasados):** {relevant_history if relevant_history else 'Ninguna.'}
**Tu Última Respuesta al Usuario:** {last_assistant_message if last_assistant_message else "Esta es la primera interacción."}
**Acción Inmediata Realizada por el Sistema:** {action_result.get('description', 'Ninguna.')}
"""
    mission = """
**Directiva Maestra:** Primero, analiza tu última respuesta y la del usuario para entender el flujo del diálogo. ¿El usuario responde a tu pregunta/sugerencia o cambia de tema? Tu respuesta debe reflejar esta comprensión para que la conversación se sienta natural. Luego, ejecuta tu misión específica.

**Misión Específica para este Turno:**
"""
    status = action_result.get('status')
    full_prompt = ""

    if status == 'SUCCESS':
        mission += """
Informa al usuario del éxito de la búsqueda. Te proporcionaré los datos crudos de las vacantes en JSON. Analízalos y preséntalos de forma útil.
**Reglas de Presentación:**
1.  Usa una tabla Markdown con las columnas "Puesto", "Empresa", "Salario".
2.  Añade una cuarta columna "Aplicar" con un enlace Markdown clicable con el texto "¡Aplica ahora!", que apunte a la `url` de la vacante.
3.  Si hay datos "primary" y "secondary", preséntalos en dos tablas separadas.
"""
        raw_data_json = json.dumps(action_result.get('data', {}), indent=2)
        full_prompt = f"{state_summary}\n{mission}\n\n**Datos Crudos para tu Respuesta:**\n```json\n{raw_data_json}\n```\n--- FIN DEL CONTEXTO ---"

    elif status in ['NO_RESULTS_FROM_API', 'NO_RELEVANT_RESULTS']:
        suggestion = get_proactive_suggestion(client, role) if role else None
        mission += f"Informa al usuario que la búsqueda no tuvo éxito. Si hay una sugerencia de rol ('{suggestion}'), úsala para proponer el siguiente paso. De lo contrario, pide otra ubicación."
        full_prompt = f"{state_summary}\n{mission}\n--- FIN DEL CONTEXTO ---"

    else: # Caso de chat general o información faltante
        if not role:
            mission += "Tu prioridad es obtener el rol que busca el usuario. Pregúntale amablemente."
        else:
            mission += "Responde al usuario de forma natural y útil, usando todo el contexto proporcionado para mantener una conversación fluida."
        full_prompt = f"{state_summary}\n{mission}\n--- FIN DEL CONTEXTO ---"

    return {"role": "system", "content": full_prompt}

def run_assistant_logic(sid: str, session: dict):
    """
    Función principal que orquesta el flujo de una respuesta del asistente.
    Esta función es ejecutada en segundo plano por cada mensaje del usuario.
    """
    try:
        # Extraer el estado actual de la sesión
        messages = session['messages']
        conversation_log = session['conversation_log']
        current_search_params = session['search_params']
        user_input = conversation_log[-1]['content']
        client = groq_create_client()

        # 1. Análisis del mensaje del usuario
        send_status_update(sid, "Entendiendo tu mensaje...", "classification")
        analysis = classify_and_extract_info(client, messages, user_input)
        logging.info(f"Análisis para {sid}: {analysis}")
        send_status_update(sid, f"Intención: {analysis['intent']}", "analysis")

        if analysis.get('role'): current_search_params['role'] = analysis.get('role')
        if analysis.get('location'): current_search_params['location'] = analysis.get('location')

        # 2. Ejecución de acciones
        action_result = {}
        role = current_search_params.get('role')
        location = current_search_params.get('location')

        if analysis.get('intent') == 'jobSearch' and role:
            action_result = fetch_and_process_vacancies(client, sid, role, location)
            location_str = f"en '{location}'" if location else "sin ubicación específica"
            action_result['description'] = f"Se realizó una búsqueda estratégica para '{role}' {location_str}."
            log_entry = {
                "type": "job_search", "timestamp": datetime.now().isoformat(),
                "details": {"role": role, "location": location, "status": action_result.get("status")}
            }
            conversation_log.append(log_entry)
        else:
            send_status_update(sid, "Pensando...", "thinking")

        # 3. Preparación del prompt para la IA
        relevant_history = find_relevant_history(client, user_input, conversation_log)
        last_assistant_message = messages[-1]['content'] if len(messages) > 1 and messages[-1]['role'] == 'assistant' else None
        mission_prompt = create_dynamic_mission_prompt(client, current_search_params, action_result, relevant_history, last_assistant_message)

        # 4. Generación y envío de la respuesta final
        send_status_update(sid, "Preparando tu respuesta...", "final_response_generation")
        request_messages = messages + [mission_prompt, {"role": "user", "content": user_input}]

        stream = client.chat.completions.create(
            messages=request_messages,
            model=MODEL_AGENT_LARGE,
            stream=True
        )

        assistant_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            assistant_response += token
            socketio.emit('assistant_token', {'token': token}, room=sid)

        # 5. Actualización del estado de la sesión
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": assistant_response})
        # No es necesario reasignar a la sesión aquí ya que estamos modificando el objeto mutable
        # session['messages'] = messages
        # session['conversation_log'] = conversation_log
        # session['search_params'] = current_search_params

        socketio.emit('assistant_response_end', {'message': assistant_response}, room=sid)
        logging.info(f"Respuesta completa enviada a {sid}.")

    except Exception as e:
        logging.error(f"Error crítico en run_assistant_logic para {sid}: {e}", exc_info=True)
        error_message = "Lo siento, ocurrió un error inesperado al procesar tu solicitud. Por favor, intenta de nuevo."
        socketio.emit('error_message', {'message': error_message}, room=sid)

# --- 5. Punto de Entrada de la Aplicación ---

if __name__ == "__main__":
    logging.info("Iniciando el servidor de Sonia...")
    # socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    socketio.run(app, debug=True, port=5000)

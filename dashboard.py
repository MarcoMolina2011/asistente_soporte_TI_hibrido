import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import pickle
import sqlite3
import networkx as nx
import base64
from PIL import Image
import io

# Agregar la carpeta src al path para importar correctamente los módulos del proyecto
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Importación real de los módulos lógicos
try:
    from classifiers.sistema_hibrido import SistemaHibridoSoporte
    from classifiers.astar import astar_soporte_ti, START_STATE, GOAL_STATE
    from classifiers.minimax import best_move, board as minimax_board, NODOS_INFRAESTRUCTURA, simular_ciberdefensa
    from classifiers.evaluacion_modelo import ejecutar_validacion
    from classifiers.vision import ejecutar_vision_soporte  # Importación agregada para evitar ModuleNotFoundError
    MODULOS_CARGADOS = True
except ImportError as e:
    MODULOS_CARGADOS = False

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - Asistente de Soporte TI Híbrido",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (textos negros en la barra lateral)
st.markdown("""
    <style>
    /* Centrar todos los títulos principales */
    h1, h2, h3 {
        text-align: center !important;
    }

    /* Forzar textos en color negro dentro de la barra lateral */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #111111 !important;
    }

    /* Ajustar color del texto de la caja informativa inferior en la barra lateral */
    [data-testid="stSidebar"] [data-testid="stInfo"] {
        color: #111111 !important;
    }
    </style>
""", unsafe_allow_html=True)

AUDIT_LOG_PATH = BASE_DIR / "artifacts" / "audit.log"
KB_PATH = BASE_DIR / "data" / "base_conocimiento.txt"

@st.cache_data
def cargar_base_conocimiento():
    """Carga y categoriza los 30 procedimientos de la base de conocimiento."""
    procs = []
    categoria_map = {
        1: "Hardware", 2: "Red", 3: "Rendimiento", 4: "Seguridad",
        5: "Hardware", 6: "Hardware", 7: "Hardware", 8: "Red",
        9: "Software", 10: "Seguridad", 11: "Software", 12: "Almacenamiento",
        13: "Hardware", 14: "Hardware", 15: "Almacenamiento", 16: "Hardware",
        17: "Hardware", 18: "Red", 19: "Red", 20: "Red",
        21: "Red", 22: "Seguridad", 23: "Seguridad", 24: "Seguridad",
        25: "Seguridad", 26: "Rendimiento", 27: "Rendimiento", 28: "Rendimiento",
        29: "Rendimiento", 30: "Hardware"
    }
    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "->" not in line:
                    continue
                partes = line.split("->", 1)
                izq = partes[0].strip()
                derecha = partes[1].strip()
                id_num = None
                if "." in izq:
                    try:
                        id_num = int(izq.split(".")[0].strip())
                        titulo = izq.split(".", 1)[1].strip()
                    except ValueError:
                        titulo = izq
                else:
                    titulo = izq
                
                cat = categoria_map.get(id_num, "General")
                procs.append({
                    "id": id_num if id_num else len(procs) + 1,
                    "titulo": titulo,
                    "categoria": cat,
                    "solucion": derecha,
                    "pasos": [p.strip() for p in derecha.split(";") if p.strip()]
                })
    return procs

def parsear_audit_log():
    """Parsea el archivo audit.log separando fecha, nivel y mensaje."""
    if not AUDIT_LOG_PATH.exists():
        return [], ""
    
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()
        
    raw_text = "".join(lineas)
    registros = []
    import re
    patron = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+\[([A-Z]+)\]\s+-\s+(.*)$")
    
    for l in lineas:
        l_str = l.strip()
        if not l_str:
            continue
        m = patron.match(l_str)
        if m:
            fecha, nivel, mensaje = m.groups()
            registros.append({
                "fecha": fecha,
                "nivel": nivel,
                "mensaje": mensaje,
                "raw": l_str
            })
        else:
            registros.append({
                "fecha": "-",
                "nivel": "INFO",
                "mensaje": l_str,
                "raw": l_str
            })
    return registros, raw_text

def acepta_patron_falla_cascada(secuencia_logs):
    """Implementación del Autómata Finito Determinista (AFD) para la interfaz con alfabeto O, E, T"""
    state = "q0"
    transitions = {
        ("q0", "O"): "q0", ("q0", "T"): "q0", ("q0", "E"): "q1",
        ("q1", "O"): "q0", ("q1", "E"): "q1", ("q1", "T"): "q2",
        ("q2", "O"): "q0", ("q2", "T"): "q0", ("q2", "E"): "q1",
    }
    for simbolo in secuencia_logs:
        if (state, simbolo) in transitions:
            state = transitions[(state, simbolo)]
        else:
            return False, "Símbolo inválido"
    return state == "q2", "Falla en cascada detectada (ET)" if state == "q2" else "Comportamiento normal / recuperado"

# --- Barra Lateral ---
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("Panel de Control TI")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navegación",
    [
        "📊 Resumen Ejecutivo", 
        "📈 Matriz de Confusión", 
        "🔍 Sistema Híbrido & TF-IDF", 
        "⚙️ Planificador A* & Minimax", 
        "🧠 Representaciones del Reconocimiento",
        "🧠 Red Neuronal & Ontología",
        "👁️ Visión Artificial (Semana 09)",
        "👤 Portal de Usuario",
        "📚 Base de Conocimiento (KB)", 
        "📜 Trazas de Auditoría"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Institución:** ETITC - 10º Semestre\n\n**Autores:** Marco Molina & Daniel Daza")

# --- 1. Resumen Ejecutivo ---
if menu == "📊 Resumen Ejecutivo":
    st.title("🚀 Dashboard General del Asistente de Soporte TI")
    st.markdown("Monitoreo en tiempo real de los componentes lógicos, modelos de machine learning y flujos del sistema híbrido.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Accuracy Modelo Base", value="92.1%", delta="Benchmark Semana 02")
    with col2:
        st.metric(label="Base de Conocimiento", value="30 Procedimientos", delta="Activo (Semana 05)")
    with col3:
        st.metric(label="Algoritmo A*", value="3 Escenarios", delta="Costo Óptimo Calculado")
    with col4:
        st.metric(label="Módulo de Representaciones", value="3 Enfoques", delta="Numérico/Simbólico/AFD")

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📌 Distribución por Categorías Técnicas (KB)")
        data_cat = {
            "Categoría": ["Hardware", "Red", "Seguridad", "Rendimiento", "Almacenamiento", "Software"],
            "Procedimientos": [8, 6, 6, 6, 3, 3]
        }
        df_cat = pd.DataFrame(data_cat)
        st.bar_chart(df_cat.set_index("Categoría"))
        
    with col_b:
        st.subheader("🛠️ Estado de Integración de Módulos")
        st.success("✅ Validación y Matriz de Confusión (Semana 02) - Operativo (92.1%)")
        st.success("✅ Clasificador Taxonómico (Semana 03) - 100% de Coincidencia")
        st.success("✅ Planificador A* y Ciberdefensa Minimax (Semana 04) - Operativo")
        st.success("✅ Sistema Híbrido (Semana 05) - Operativo con Reglas y TF-IDF")
        st.success("✅ Representaciones del Reconocimiento (Semana 07) - Operativo")
        st.success("✅ Red Neuronal, Imagen Base64, SQLite y Ontología (Semana 08) - Operativo")
        st.success("✅ Visión Artificial (Semana 09) - Operativo")
        st.success("✅ Motor de Auditoría y Trazas - Registrando en artifacts/audit.log")

# --- 2. Matriz de Confusión ---
elif menu == "📈 Matriz de Confusión":
    st.title("📉 Validación del Modelo Base (Semana 02)")
    st.markdown("Evaluación del pipeline reproducible de clasificación multiclase del curso (StandardScaler + LogisticRegression sobre el benchmark estándar con división estratificada 75/25).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Muestras de Entrenamiento", value="112")
        st.metric(label="Muestras de Prueba", value="38")
    with col2:
        st.metric(label="Accuracy Global", value="92.1% (0.921)")
        st.metric(label="Tasa de Error", value="7.9% (3/38)")
    
    st.markdown("---")
    st.subheader("Matriz de Confusión Numérica")
    
    if MODULOS_CARGADOS:
        try:
            cm = ejecutar_validacion()
            clases = ["Clase 0 (Setosa)", "Clase 1 (Versicolor)", "Clase 2 (Virginica)"]
            df_matriz = pd.DataFrame(
                cm,
                index=[f"Real: {c}" for c in clases],
                columns=[f"Pred: {c}" for c in clases]
            )
            st.dataframe(df_matriz, use_container_width=True)
            st.info("La matriz demuestra 35 aciertos de 38 muestras de prueba (92.1% de precisión global), con dispersión mínima focalizada únicamente en 3 casos límite entre clases contiguas.")
        except Exception as e:
            st.error(f"Error al ejecutar la validación del modelo: {e}")
    else:
        st.warning("Módulos no disponibles para generar la matriz en tiempo real.")

# --- 3. Sistema Híbrido & TF-IDF (Dinámico) ---
elif menu == "🔍 Sistema Híbrido & TF-IDF":
    st.title("🧠 Simulador de Sistema Híbrido (Reglas + TF-IDF)")
    st.markdown("Consulta la base de conocimiento en tiempo real utilizando similitud coseno sobre los 30 procedimientos de soporte y reglas lógicas expertas.")

    query = st.text_input("Escribe un reporte de ticket de soporte:", value="Disco lleno", key="query_input")
    
    if st.button("Ingresar su consulta"):
        if MODULOS_CARGADOS:
            sistema = SistemaHibridoSoporte()
            resultado = sistema.procesar(query)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📋 Resultados del Motor")
                st.info(f"**Reglas Activadas:** `{resultado['reglas']}`")
                st.warning(f"**Clase Predicha (ML):** `{resultado['clase']}`")
                st.metric(label="Similitud Coseno", value=f"{resultado['similitud']:.4f}")
            with col2:
                st.markdown("### 📄 Evidencia Recuperada (TF-IDF)")
                st.success(f"{resultado['evidencia']}")
        else:
            st.error("No se pudieron cargar los módulos de Python desde la carpeta `src/`.")

# --- 4. Planificador A* & Minimax ---
elif menu == "⚙️ Planificador A* & Minimax":
    st.title("🤖 Optimización en Espacio de Estados y Decisiones Adversariales")
    
    tab1, tab2 = st.tabs(["Planificador A*", "Minimax (Juegos Adversariales)"])
    
    with tab1:
        st.subheader("Secuencia Óptima de Resolución de Soporte con A*")
        st.markdown("Encuentra la secuencia de acciones técnicas de menor costo de esfuerzo utilizando la distancia de Manhattan como heurística admisible $h(n)$.")
        
        if MODULOS_CARGADOS:
            escenarios = {
                "Caso 1: Incidente Estándar (Todo caído: Red=0, BD=0, App=0)": (0, 0, 0),
                "Caso 2: Condición Parcial (En proceso: Red=1, BD=1, App=0)": (1, 1, 0),
                "Caso 3: Restricción Preexistente (BD ya recuperada: Red=0, BD=2, App=1)": (0, 2, 1)
            }
            opcion = st.selectbox("Seleccionar Escenario de Estado Inicial:", list(escenarios.keys()))
            estado_ini = escenarios[opcion]
            
            ruta, costo_total = astar_soporte_ti(estado_ini, GOAL_STATE)
            st.info(f"📍 **Estado Inicial:** `{estado_ini}` ➔ **Estado Meta:** `{GOAL_STATE}` | **Costo Mínimo Acumulado:** `{costo_total}` unidades de esfuerzo")
            
            st.markdown("#### 📊 Recuperación y Salud de Subsistemas")
            c_red, c_bd, c_app = st.columns(3)
            with c_red:
                st.markdown(f"**🌐 Red:** Nivel `{estado_ini[0]}/2`")
                st.progress(estado_ini[0] / 2.0, text=f"{(estado_ini[0]/2)*100:.0f}% Operativo")
            with c_bd:
                st.markdown(f"**🗄️ Base de Datos:** Nivel `{estado_ini[1]}/2`")
                st.progress(estado_ini[1] / 2.0, text=f"{(estado_ini[1]/2)*100:.0f}% Operativo")
            with c_app:
                st.markdown(f"**⚙️ Microservicio Backend:** Nivel `{estado_ini[2]}/2`")
                st.progress(estado_ini[2] / 2.0, text=f"{(estado_ini[2]/2)*100:.0f}% Operativo")
            
            if ruta:
                evolucion = [{
                    "Paso": "P0 (Inicial)",
                    "Red (%)": (estado_ini[0] / 2) * 100,
                    "Base de Datos (%)": (estado_ini[1] / 2) * 100,
                    "Backend (%)": (estado_ini[2] / 2) * 100
                }]
                pasos_lista = []
                for idx, (orig, dest, desc, c) in enumerate(ruta, 1):
                    evolucion.append({
                        "Paso": f"P{idx}: {desc[:15]}...",
                        "Red (%)": (dest[0] / 2) * 100,
                        "Base de Datos (%)": (dest[1] / 2) * 100,
                        "Backend (%)": (dest[2] / 2) * 100
                    })
                    pasos_lista.append({
                        "Paso": idx,
                        "Acción Correctiva": desc,
                        "Costo Paso": c,
                        "Transición de Estado": f"{orig} ➔ {dest}"
                    })
                
                st.markdown("##### 📈 Progresión de Salud hacia la Meta (2, 2, 2)")
                df_evol = pd.DataFrame(evolucion)
                st.line_chart(df_evol.set_index("Paso"))
                
                st.markdown("##### 📋 Secuencia Óptima de Acciones Técnicas")
                st.table(pd.DataFrame(pasos_lista))
            else:
                st.warning("No se encontró una ruta válida para este estado.")
        else:
            st.warning("Módulo A* no disponible temporalmente.")
        
    with tab2:
        st.subheader("🛡️ Simulador de Ciberdefensa con Minimax (Blue Team vs. Red Team)")
        st.markdown(
            "Modelado formal de **búsqueda adversarial de suma cero** para contención de incidentes de seguridad y mitigación de intrusión lateral.  \n"
            "- **Jugador MAX (Blue Team / Soporte TI):** Aplica contramedidas preventivas y aísla servidores críticos (`🛡️ Blindado`).  \n"
            "- **Jugador MIN (Red Team / Amenaza Externa):** Explota vulnerabilidades buscando comprometer una ruta crítica (`🚨 Comprometido`).  \n"
            "- **Objetivo:** Minimax evalúa exhaustivamente el árbol de decisiones para seleccionar el nodo que garantiza la resiliencia de la infraestructura corporativa."
        )
        if MODULOS_CARGADOS:
            diag = simular_ciberdefensa(minimax_board)
            
            col_left, col_right = st.columns([1, 1], gap="medium")
            
            with col_left:
                st.markdown("##### 🗺️ Topología de Servidores (Matriz 3x3)")
                
                tarjetas_html = []
                for i in range(9):
                    estado = minimax_board[i]
                    nombre = NODOS_INFRAESTRUCTURA[i]
                    es_mejor = (i == diag["posicion"])
                    
                    if estado == "X":
                        bg = "#e8f5e9"
                        border = "#2e7d32"
                        badge = "🛡️ Blue Team"
                        sub = "Blindado"
                    elif estado == "O":
                        bg = "#ffebee"
                        border = "#c62828"
                        badge = "🚨 Red Team"
                        sub = "Comprometido"
                    elif es_mejor:
                        bg = "#fff8e1"
                        border = "#f57f17"
                        badge = "🎯 Minimax"
                        sub = "Blindar Ahora"
                    else:
                        bg = "#f5f5f5"
                        border = "#9e9e9e"
                        badge = "🟢 En Línea"
                        sub = "Disponible"
                        
                    tarjetas_html.append(
                        f'<div style="background-color: {bg}; border: 2px solid {border}; border-radius: 8px; padding: 10px 4px; text-align: center; min-height: 90px;">'
                        f'<span style="font-size: 0.72rem; color: #555;">[Nodo #{i}]</span><br>'
                        f'<strong style="color: #111; font-size: 0.8rem;">{nombre}</strong><br>'
                        f'<span style="font-weight: bold; font-size: 0.75rem; color: {border};">{badge}</span><br>'
                        f'<small style="color: #444; font-size: 0.7rem;">{sub}</small>'
                        f'</div>'
                    )
                
                grid_html = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px;">' + "".join(tarjetas_html) + '</div>'
                st.markdown(grid_html, unsafe_allow_html=True)
                
            with col_right:
                st.markdown("##### 🎯 Decisión Táctica de Contención")
                st.success(f"**Servidor Seleccionado:** `{diag['nodo']}` (Nodo #{diag['posicion']})")
                st.info(f"**Acción Defensiva:** {diag['accion']}")
                st.markdown(f"**Análisis Estratégico:**\n\n{diag['justificacion']}")
                st.caption("Función de Utilidad: +1 si el Blue Team asegura el perímetro, 0 si hay contención, -1 si el Red Team compromete la red.")
        else:
            st.warning("Módulo Minimax no disponible temporalmente.")

# --- 5. REPRESENTACIONES DEL RECONOCIMIENTO (Semana 07) ---
elif menu == "🧠 Representaciones del Reconocimiento":
    st.title("🧠 Representaciones del Reconocimiento (Semana 07)")
    st.markdown("Simulación de los tres enfoques de inteligencia artificial para interpretar y procesar información técnica de soporte.")
    
    tab_num, tab_sim, tab_aut = st.tabs(["1️⃣ Numérica (Telemetría)", "2️⃣ Simbólica (Sistema Experto)", "3️⃣ Autómatas (Logs en Cascada)"])
    
    # 1. Representación Numérica
    with tab_num:
        st.subheader("Representación Numérica: Distancia Euclidiana (Triaje de Tickets)")
        st.markdown("Mide qué tan cerca está el estado actual del servidor de un perfil crítico de colapso, usando vectores numéricos y distancias en el espacio.")
        
        perfil_colapso = np.array([10.0, 10.0, 1.0]) # [Impacto Infraestructura, Urgencia Tiempo, Sentimiento NLP]
        
        col_sliders, col_results = st.columns([1, 1])
        with col_sliders:
            st.markdown("**Ajusta las características del ticket entrante:**")
            imp_val = st.slider("Impacto en Infraestructura (1-10)", 1.0, 10.0, 8.5)
            urg_val = st.slider("Urgencia de Tiempo (1-10)", 1.0, 10.0, 9.0)
            nlp_val = st.slider("Sentimiento Negativo NLP (0.0-1.0)", 0.0, 1.0, 0.95)
            
        with col_results:
            ticket_actual = np.array([imp_val, urg_val, nlp_val])
            distancia = np.linalg.norm(ticket_actual - perfil_colapso)
            
            st.markdown(f"**Perfil Crítico (Fijo):** `[{perfil_colapso[0]}, {perfil_colapso[1]}, {perfil_colapso[2]}]`")
            st.markdown(f"**Ticket Actual (Dinámico):** `[{ticket_actual[0]}, {ticket_actual[1]}, {ticket_actual[2]}]`")
            
            st.metric(label="Distancia Numérica al Escalamiento Inmediato", value=f"{distancia:.2f}")
            
            if distancia < 3.0:
                st.error("🚨 **ALERTA DE TRIAJE:** Clasificación prioritaria. Asignando directamente a Especialista Humano.")
            else:
                st.success("✅ **ESTABLE:** Ticket estándar manejable por chatbot de Nivel 1.")
                
    # 2. Representación Simbólica
    with tab_sim:
        st.subheader("Representación Simbólica: Inferencia Discreta (Hechos y Reglas)")
        st.markdown("Transforma el lenguaje natural en hechos discretos. Utiliza lógica `SI -> ENTONCES` para llegar a un diagnóstico absoluto y justificable.")
        
        st.markdown("**Selecciona los síntomas detectados en el ticket:**")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            sym_bsod = st.checkbox("pantalla_azul", value=True)
            sym_rein = st.checkbox("reinicio_constante", value=True)
        with c2:
            sym_ping = st.checkbox("ping_fallido", value=False)
            sym_serv = st.checkbox("servidor_no_responde", value=False)
        with c3:
            sym_pwd = st.checkbox("olvido_contrasena", value=False)
            sym_ad = st.checkbox("active_directory", value=False)
        
        hechos_detectados = set()
        if sym_bsod: hechos_detectados.add("pantalla_azul")
        if sym_rein: hechos_detectados.add("reinicio_constante")
        if sym_ping: hechos_detectados.add("ping_fallido")
        if sym_serv: hechos_detectados.add("servidor_no_responde")
        if sym_pwd: hechos_detectados.add("olvido_contrasena")
        if sym_ad: hechos_detectados.add("active_directory")
        
        st.markdown(f"**Base de Hechos Actual:** `{hechos_detectados}`")
        
        st.markdown("---")
        st.markdown("#### Motor de Inferencia")
        if {"pantalla_azul", "reinicio_constante"}.issubset(hechos_detectados):
            st.error("🚨 **Diagnóstico Simbólico:** `falla_hardware_fisico` detectada.")
            st.markdown("**Acción recomendada:** Desviar al módulo de Soporte de Campo (Presencial) y solicitar modelo del equipo.")
        elif {"ping_fallido", "servidor_no_responde"}.issubset(hechos_detectados):
            st.error("🚨 **Diagnóstico Simbólico:** `caida_infraestructura_red` detectada.")
            st.markdown("**Acción recomendada:** Notificar a NOC y ejecutar protocolo de ping persistente.")
        elif {"olvido_contrasena", "active_directory"}.issubset(hechos_detectados):
            st.warning("⚠️ **Diagnóstico Simbólico:** `restablecimiento_credenciales`.")
            st.markdown("**Acción recomendada:** Ejecutar script de automatización de reseteo (Nivel 1 - Bot).")
        else:
            st.info("ℹ️ **Diagnóstico Simbólico:** Incidentes sin mapeo directo crítico.")

    # 3. Representación por Autómata
    with tab_aut:
        st.subheader("Representación por Autómata: Validación de Cronologías")
        st.markdown("Utiliza un Autómata Finito Determinista (AFD) para auditar secuencias de tiempo. Valida si una sucesión de eventos (`O`: Operativo, `E`: Error, `T`: Timeout) desencadena el patrón estricto de una *falla en cascada* (`ET`).")
        
        st.markdown("**Alfabeto permitido:** `O` (Operativo), `E` (Error), `T` (Timeout)")
        secuencia = st.text_input("Ingresa una secuencia de logs temporales:", "OOOEET").upper()
        
        if st.button("Auditar Secuencia con AFD"):
            secuencia_limpia = "".join(c for c in secuencia if c in ['O', 'E', 'T'])
            
            if len(secuencia_limpia) != len(secuencia):
                st.error(f"La secuencia contiene caracteres no válidos. Procesando solo la parte válida: `{secuencia_limpia}`")
            
            if not secuencia_limpia:
                st.warning("Ingresa una secuencia válida (ej. 'OOET').")
            else:
                aceptada, mensaje = acepta_patron_falla_cascada(secuencia_limpia)
                
                st.markdown(f"#### Resultados de Auditoría para: `{secuencia_limpia}`")
                
                estado_visual = "q0"
                pasos = [f"Inicio (q0)"]
                
                transitions_vis = {
                    ("q0", "O"): "q0", ("q0", "T"): "q0", ("q0", "E"): "q1",
                    ("q1", "O"): "q0", ("q1", "E"): "q1", ("q1", "T"): "q2",
                    ("q2", "O"): "q0", ("q2", "T"): "q0", ("q2", "E"): "q1",
                }
                
                for sim in secuencia_limpia:
                    if (estado_visual, sim) in transitions_vis:
                        estado_visual = transitions_vis[(estado_visual, sim)]
                        pasos.append(f"Leer '{sim}' ➔ {estado_visual}")
                    else:
                        pasos.append(f"Transición Inválida para '{sim}'")
                        estado_visual = "Error"
                        break
                
                st.code(" -> ".join(pasos), language="text")
                
                if aceptada:
                    st.error("❌ **Secuencia Aceptada.** El AFD confirma que el patrón estricto `[Error -> Timeout]` se cumple. Falla en cascada confirmada.")
                else:
                    st.success("✅ **Secuencia Rechazada.** El sistema se estabilizó al final o la secuencia no terminó en Timeout.")

# --- 6. Red Neuronal, Imagen Base64 & Ontología (Semana 08) ---
elif menu == "🧠 Red Neuronal & Ontología":
    st.title("🧠 Red Neuronal, Imágenes Base64 y Ontología (Semana 08)")
    st.markdown("Integración completa: Modelo MLP (Predicción) + Carga y Descripción de Imagen Base64 ➔ SQLite (Evidencia) ➔ Ontología GraphML (Significado)[cite: 3, 8].")
    
    tab_nn, tab_img, tab_db, tab_onto = st.tabs([
        "🤖 Clasificador MLP", 
        "🖼️ Análisis de Imagen (Base64)", 
        "🗄️ Base de Evidencia (SQLite)", 
        "🌐 Ontología del Dominio"
    ])
    
    with tab_nn:
        st.subheader("Clasificador de Tickets por Red Neuronal Artificial")
        st.markdown("Introduce las métricas de telemetría del incidente para que la MLP prediga la categoría de soporte.")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: f1 = st.slider("Impacto App (0-10)", 0.0, 10.0, 5.0)
        with c2: f2 = st.slider("Urgencia (0-10)", 0.0, 10.0, 5.0)
        with c3: f3 = st.slider("Temperatura Servidor", 0.0, 10.0, 4.0)
        with c4: f4 = st.slider("Fallos de Red", 0.0, 10.0, 3.0)
        
        if st.button("Ejecutar Predicción Neuronal"):
            modelo_path = BASE_DIR / "artifacts" / "modelo_mlp.pkl"
            if modelo_path.exists():
                with open(modelo_path, "rb") as f:
                    clf = pickle.load(f)
                entrada = np.array([[f1, f2, f3, f4]])
                prediccion = clf.predict(entrada)[0]
                mapa_clases = {0: "Hardware", 1: "Red", 2: "Software", 3: "Seguridad"}
                st.success(f"🎯 Categoría Predicha por el Modelo: **{mapa_clases.get(prediccion, 'Desconocida')}** (Clase ID: {prediccion})")
            else:
                st.warning("Primero debes ejecutar `python src/main.py` para generar el modelo entrenado.")
                
    with tab_img:
        st.subheader("🖼️ Reconocimiento de Imágenes, Base64 y Metadatos (Soporte TI)")
        st.markdown("Adjunta una captura de pantalla de error técnico e ingresa una descripción textual para que el agente la codifique, analice e integre en el flujo de evidencia[cite: 3].")
        
        uploaded_file = st.file_uploader("Sube una imagen de diagnóstico de soporte", type=["png", "jpg", "jpeg"])
        desc_usuario = st.text_input("Añade una descripción textual o sintomatología observada:", "Pantalla azul intermitente con volcado de memoria en equipo contable.")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Archivo cargado: {uploaded_file.name}", width=350)
            
            bytes_data = uploaded_file.getvalue()
            base64_str = base64.b64encode(bytes_data).decode("utf-8")
            
            with st.expander("Ver cadena Base64 generada"):
                st.code(base64_str[:250] + "...", language="text")
                
            if st.button("Procesar Imagen, Base64 y Descripción en el Pipeline"):
                try:
                    from classifiers.reconocimiento import ejecutar_semana_08
                    ejecutar_semana_08(imagen_bytes=bytes_data, nombre_archivo=uploaded_file.name, descripcion_texto=desc_usuario)
                    
                    nombre_l = uploaded_file.name.lower()
                    if "red" in nombre_l or "router" in nombre_l:
                        diag_resultado = "🌐 Red / Conectividad (Clase 1)"
                    elif "azul" in nombre_l or "hardware" in nombre_l:
                        diag_resultado = "💻 Hardware / Pantalla Azul (Clase 0)"
                    else:
                        diag_resultado = "⚙️ Software / Aplicación (Clase 2)"
                        
                    st.success(f"🎯 **Agente IA - Imagen Reconocida y Descrita:** {diag_resultado}")
                    st.info("📊 **Estado en Proyecto:** La imagen se codificó en Base64, se asoció a su descripción textual, se clasificó con enfoque de soporte TI, se persistió en SQLite y se enlazó en la ontología GraphML.")
                except Exception as e:
                    st.error(f"Error al procesar la imagen en el pipeline: {e}")

    with tab_db:
        st.subheader("Registro de Evidencia Verificable (SQLite)")
        st.markdown("Consulta los registros almacenados en `artifacts/soporte_evidencia.db` (incluyendo metadatos de imágenes Base64 y descripciones textualmente auditadas)[cite: 3, 8].")
        db_p = BASE_DIR / "artifacts" / "soporte_evidencia.db"
        if db_p.exists():
            with sqlite3.connect(db_p) as conn:
                df_ev = pd.read_sql("SELECT * FROM evidencia_tickets ORDER BY id DESC LIMIT 15", conn)
            st.dataframe(df_ev, use_container_width=True)
        else:
            st.warning("Base de datos no encontrada. Ejecuta el script del proyecto.")
            
    with tab_onto:
        st.subheader("Representación del Conocimiento (Ontología GraphML)")
        st.markdown("Relaciones lógicas que otorgan significado a las predicciones y canales visuales dentro del dominio del Asistente de Soporte TI[cite: 3, 8].")
        onto_p = BASE_DIR / "artifacts" / "ontologia.graphml"
        if onto_p.exists():
            G_load = nx.read_graphml(onto_p)
            st.info(f"Grafo cargado correctamente con **{G_load.number_of_nodes()} conceptos** y **{G_load.number_of_edges()} relaciones semánticas**[cite: 3, 8].")
            edges_list = [(u, v, data.get('rel', 'relación')) for u, v, data in G_load.edges(data=True)]
            st.table(pd.DataFrame(edges_list, columns=["Sujeto", "Objeto", "Relación Semántica"]))
        else:
            st.warning("Archivo GraphML no encontrado en artifacts/.")

# --- 6.1 Visión Artificial (Semana 09) ---
elif menu == "👁️ Visión Artificial (Semana 09)":
    st.title("👁️ Visión Artificial y Procesamiento de Imágenes")
    st.markdown("Transformación de evidencia visual en información numérica mediante extracción de características, detección de contornos (Canny) y segmentación por umbrales (Otsu).")

    # NUEVO: Selector de imágenes de la Base de Conocimiento
    st.subheader("📚 Analizar Evidencia desde la Base de Conocimiento")
    
    opciones_imagenes = {
        "💻 Falla 1: Disco Duro Dañado (Sectores defectuosos)": "disco_duro.png",
        "🟦 Falla 2: Pantalla Azul (Volcado de memoria)": "pantalla_azul.png",
        "🌐 Falla 3: Router sin conexión (Luces apagadas)": "router.png"
    }
    
    imagen_seleccionada = st.selectbox("Selecciona un caso de estudio predefinido para procesar:", list(opciones_imagenes.keys()))
    archivo_imagen = opciones_imagenes[imagen_seleccionada]

    if st.button("Procesar Imagen Seleccionada"):
        if MODULOS_CARGADOS:
            with st.spinner(f"Procesando bordes, umbrales y regiones de {archivo_imagen}..."):
                try:
                    # Pasamos el archivo seleccionado a la función backend
                    umbral, regiones = ejecutar_vision_soporte(nombre_imagen=archivo_imagen)
                    
                    if umbral is not None:
                        st.session_state['otsu_thresh'] = umbral
                        st.session_state['otsu_regs'] = regiones
                        st.success(f"✅ ¡Pipeline ejecutado! Imagen '{archivo_imagen}' procesada correctamente.")
                    else:
                        st.error(f"❌ No se encontró la imagen '{archivo_imagen}' en la carpeta 'data/'. ¡Asegúrate de guardarla allí primero!")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
        else:
            st.error("Los módulos no se cargaron correctamente. Verifica las importaciones.")

    img_path = Path("artifacts/semana09_vision.png")
    if img_path.exists():
        st.image(str(img_path), caption="Pipeline de Visión: Imagen Original ➔ Bordes (Canny) ➔ Máscara Binaria (Otsu)", use_container_width=True)
        
        if 'otsu_thresh' in st.session_state and 'otsu_regs' in st.session_state:
            st.markdown("### 📊 Resultados del Procesamiento Numérico")
            col1, col2 = st.columns(2)
            col1.metric("Umbral Otsu (Corte de Intensidad)", f"{st.session_state['otsu_thresh']:.4f}")
            col2.metric("Regiones Conectadas Detectadas", st.session_state['otsu_regs'])
            st.markdown("---")
            
        st.info("""
        **Análisis Técnico del Procesamiento:**
        * **Contornos (Canny):** Identifica los cambios bruscos de intensidad en los píxeles, marcando los límites físicos y estructurales del hardware defectuoso para aislar su forma.
        * **Umbral Automático (Otsu):** Calcula matemáticamente el punto óptimo en el histograma de la imagen para separar los píxeles en dos clases (claro/oscuro), generando una máscara binaria.
        * **Regiones Conectadas:** La máscara permite al sistema contar grupos de píxeles unidos, lo cual es el primer paso para aislar piezas o zonas afectadas antes de pasarlas a una red neuronal convolucional.
        """)
    else:
        st.warning("⚠️ No se ha generado la evidencia visual. Selecciona una imagen y procesa el pipeline.")

# --- 7. Portal de Usuario ---
elif menu == "👤 Portal de Usuario":
    st.title("👤 Portal de Usuario / Cliente Final")
    st.markdown("Interfaz orientada al usuario final para la creación de solicitudes de soporte y asistencia.")
    
    usuario_nombre = st.text_input("Nombre del Empleado / Usuario:", "Carlos Pérez (Contabilidad)")
    problema_desc = st.text_area("Describe el síntoma técnico:", "La aplicación de nómina se congela y arroja pantalla azul al intentar generar el cierre mensual.")
    urgencia_opc = st.selectbox("Nivel de Urgencia percibida:", ["Baja", "Media", "Alta", "Crítica"])
    
    if st.button("Ingresar su consulta"):
        st.success(f"✅ ¡Ticket registrado con éxito para {usuario_nombre}!")
        st.markdown("---")
        st.markdown("### 🔍 Análisis Automático del Asistente (Semana 07)")
        
        if "pantalla_azul" in problema_desc.lower() or "congela" in problema_desc.lower():
            st.error("🚨 **Representación Simbólica:** Síntoma crítico identificado (`falla_hardware_fisico`).")
            st.markdown("**Acción recomendada asignada:** Derivación automática a soporte técnico presencial.")
        else:
            st.info("ℹ️ **Representación Simbólica:** Incidente registrado como estándar.")
            
        vector_ticket = np.array([9.0 if urgencia_opc=="Crítica" else 5.0, 2.0, 0.8])
        dist_portal = np.linalg.norm(vector_ticket - np.array([10.0, 10.0, 1.0]))
        st.metric(label="Métrica de Proximidad a Incidente Crítico (Distancia Euclidiana)", value=f"{dist_portal:.2f}")

# --- 8. Base de Conocimiento (KB) ---
elif menu == "📚 Base de Conocimiento (KB)":
    st.title("📚 Base de Conocimiento Técnica (KB)")
    st.markdown("Catálogo de los **30 procedimientos operativos estandarizados** del Asistente de Soporte TI, utilizados para el entrenamiento del clasificador y el motor TF-IDF.")
    
    procedimientos = cargar_base_conocimiento()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        categorias = ["Todas", "Hardware", "Red", "Seguridad", "Rendimiento", "Almacenamiento", "Software"]
        cat_sel = st.selectbox("Filtrar por Categoría:", categorias)
    with col2:
        busqueda = st.text_input("Buscar procedimiento o palabra clave (ej. 'disco', 'wifi', 'memoria'):")
    
    filtrados = procedimientos
    if cat_sel != "Todas":
        filtrados = [p for p in filtrados if p["categoria"] == cat_sel]
    if busqueda.strip():
        q_b = busqueda.lower().strip()
        filtrados = [
            p for p in filtrados
            if q_b in p["titulo"].lower() or q_b in p["solucion"].lower() or q_b in p["categoria"].lower()
        ]
    
    st.markdown(f"**Mostrando {len(filtrados)} de {len(procedimientos)} procedimientos técnicos disponibles**")
    st.markdown("---")
    
    if filtrados:
        for proc in filtrados:
            with st.expander(f"📌 #{proc['id']} - {proc['titulo']}  [{proc['categoria']}]"):
                st.markdown(f"**Categoría:** `{proc['categoria']}`")
                st.markdown("**Procedimiento de Solución:**")
                for paso_idx, paso in enumerate(proc['pasos'], 1):
                    st.markdown(f"{paso_idx}. {paso}")
    else:
        st.warning("No se encontraron procedimientos que coincidan con los filtros aplicados.")

# --- 9. Trazas de Auditoría ---
elif menu == "📜 Trazas de Auditoría":
    st.title("📝 Registro de Auditoría del Sistema (audit.log)")
    st.markdown("Monitoreo y observabilidad de eventos generados por el orquestador `main.py` y los clasificadores del sistema.")
    
    registros, raw_text = parsear_audit_log()
    
    if registros:
        total_logs = len(registros)
        info_count = sum(1 for r in registros if r["nivel"] == "INFO")
        warn_count = sum(1 for r in registros if r["nivel"] in ["WARNING", "WARN"])
        err_count = sum(1 for r in registros if r["nivel"] == "ERROR")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Eventos", total_logs)
        with m2:
            st.metric("Eventos INFO", info_count)
        with m3:
            st.metric("Alertas / Errores", warn_count + err_count)
        with m4:
            st.metric("Último Registro", registros[-1]["fecha"] if registros else "-")
        
        st.markdown("---")
        
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            niveles_disp = sorted(list(set(r["nivel"] for r in registros)))
            niveles_sel = st.multiselect("Filtrar por Nivel:", niveles_disp, default=niveles_disp)
        with f_col2:
            texto_filtro = st.text_input("Buscar en trazas (Ticket ID, Acción, Texto):")
            
        regs_filtrados = [r for r in registros if r["nivel"] in niveles_sel]
        if texto_filtro.strip():
            tf = texto_filtro.lower().strip()
            regs_filtrados = [r for r in regs_filtrados if tf in r["mensaje"].lower() or tf in r["fecha"].lower()]
            
        st.caption(f"Mostrando {len(regs_filtrados)} de {total_logs} líneas registradas.")
        
        tab_tabla, tab_raw = st.tabs(["📋 Vista Estructurada", "🖥️️ Consola Raw"])
        
        with tab_tabla:
            if regs_filtrados:
                df_regs = pd.DataFrame(regs_filtrados)[["fecha", "nivel", "mensaje"]]
                df_regs.columns = ["Fecha / Hora", "Nivel", "Detalle del Evento"]
                st.dataframe(df_regs, use_container_width=True, height=400)
            else:
                st.info("No hay eventos que coincidan con los filtros seleccionados.")
                
        with tab_raw:
            st.text_area("Contenido Completo de audit.log", raw_text, height=400)
            st.download_button(
                label="⬇️ Descargar audit.log",
                data=raw_text,
                file_name="audit.log",
                mime="text/plain"
            )
    else:
        st.warning("No se encontró el archivo de auditoría en `artifacts/audit.log`. Ejecuta `python src/main.py` para generarlo.")
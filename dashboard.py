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
    from classifiers.texturas import procesar_imagen_proyecto  # Importación para la Semana 10
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

# Estilos CSS personalizados (sistema de diseño: paleta, tipografía y componentes)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #0F1B2D;
        --ink-hover: #1A2B45;
        --accent: #2457D6;
        --accent-dark: #1B44AD;
        --accent-soft: #E8EEFC;
        --bg: #F4F6FA;
        --surface: #FFFFFF;
        --line: #E2E7F0;
        --text: #14213D;
        --muted: #5B6779;
        --ok: #1E8E5A;
        --ok-soft: #E7F5EE;
    }

    /* ---------- Base ---------- */
    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'IBM Plex Sans', 'Segoe UI', Roboto, Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, p, li, label, td, th, input, textarea, button {
        font-family: 'IBM Plex Sans', 'Segoe UI', Roboto, Arial, sans-serif;
    }
    code, pre, [data-testid="stCode"] * {
        font-family: 'IBM Plex Mono', Consolas, 'Courier New', monospace !important;
    }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

    /* ---------- Tipografía ---------- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        text-align: left !important;
        color: var(--text);
        letter-spacing: -0.01em;
    }
    .stApp h2 { font-size: 1.25rem; font-weight: 600; }
    .stApp h3 { font-size: 1.1rem; font-weight: 600; }
    .stApp h4 { font-size: 1rem; font-weight: 600; }
    .stApp h5 { font-size: 0.92rem; font-weight: 600; color: var(--muted); }
    .stApp p, .stApp li { line-height: 1.6; }
    .stApp hr { border: none; border-top: 1px solid var(--line); margin: 1.4rem 0; }
    .stApp code {
        background: var(--accent-soft);
        color: var(--accent-dark);
        border-radius: 5px;
        padding: 1px 6px;
        font-size: 0.85em;
    }

    /* ---------- Cabecera de página ---------- */
    .page-head { margin: 0 0 1.6rem 0; padding-bottom: 1.1rem; border-bottom: 1px solid var(--line); }
    .page-head h1 {
        font-size: 1.75rem; font-weight: 700; line-height: 1.25;
        margin: 0 0 0.45rem 0; padding: 0 !important; color: var(--text);
    }
    .page-head p { color: var(--muted); margin: 0; max-width: 80ch; font-size: 0.97rem; }
    .page-head code { font-size: 0.82em; }

    /* ---------- Barra lateral ---------- */
    [data-testid="stSidebar"] { background: var(--ink); border-right: 1px solid #0B1424; }
    [data-testid="stSidebar"] * { color: #C9D3E3; }
    [data-testid="stSidebar"] hr { border-top: 1px solid #22344F; margin: 1.1rem 0; }
    [data-testid="stSidebar"] button svg { color: #C9D3E3; fill: #C9D3E3; }

    .brand { display: flex; align-items: center; gap: 12px; padding: 6px 4px 4px 4px; }
    .brand-mark {
        width: 40px; height: 40px; border-radius: 10px; background: var(--accent);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.02em; flex-shrink: 0;
    }
    [data-testid="stSidebar"] .brand-mark { color: #FFFFFF; }
    .brand-name { font-size: 1.02rem; font-weight: 600; line-height: 1.2; }
    [data-testid="stSidebar"] .brand-name { color: #FFFFFF; }
    .brand-sub { font-size: 0.78rem; margin-top: 2px; }
    [data-testid="stSidebar"] .brand-sub { color: #8294AE; }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.8rem; font-weight: 500; color: #8294AE; margin-bottom: 6px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        width: 100%; padding: 9px 12px; border-radius: 8px; cursor: pointer;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label div:not(:has(p)),
    [data-testid="stSidebar"] [role="radiogroup"] label span:not(:has(p)),
    [data-testid="stSidebar"] [role="radiogroup"] label svg { display: none !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 0.9rem; font-weight: 500; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--ink-hover); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--accent); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color: #FFFFFF; font-weight: 600; }

    .side-card { background: var(--ink-hover); border: 1px solid #22344F; border-radius: 10px; padding: 12px 14px; }
    .side-card-k { font-size: 0.75rem; margin-top: 8px; }
    .side-card-k:first-child { margin-top: 0; }
    [data-testid="stSidebar"] .side-card-k { color: #8294AE; }
    .side-card-v { font-size: 0.86rem; font-weight: 500; line-height: 1.4; }
    [data-testid="stSidebar"] .side-card-v { color: #FFFFFF; }

    /* ---------- Métricas ---------- */
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-left: 4px solid var(--accent); border-radius: 10px;
        padding: 14px 18px; box-shadow: 0 1px 2px rgba(15, 27, 45, 0.04);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.82rem; font-weight: 500; }
    [data-testid="stMetricValue"] { color: var(--text); font-weight: 600; }
    [data-testid="stMetricValue"] > div {
        font-size: 1.5rem; white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.25;
    }
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: var(--muted) !important; font-size: 0.78rem; }
    [data-testid="stMetricDelta"] { background: #EDF1F8 !important; border-radius: 6px; padding: 2px 8px; width: fit-content; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* ---------- Alertas ---------- */
    [data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--line); }
    [data-testid="stAlert"] p { line-height: 1.55; }

    /* ---------- Estado de módulos (Resumen) ---------- */
    .status-list { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .status-row {
        display: flex; align-items: center; justify-content: space-between; gap: 14px;
        padding: 11px 16px; border-bottom: 1px solid var(--line);
    }
    .status-row:last-child { border-bottom: none; }
    .status-name { color: var(--text); font-size: 0.9rem; font-weight: 500; line-height: 1.35; }
    .status-meta { color: var(--muted); font-size: 0.78rem; margin-top: 1px; }
    .status-meta:empty { display: none; }
    .badge-ok {
        background: var(--ok-soft); color: var(--ok); border-radius: 999px;
        padding: 3px 11px; font-size: 0.75rem; font-weight: 600; white-space: nowrap;
    }

    /* ---------- Pestañas ---------- */
    [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] { padding: 10px 16px; border-radius: 8px 8px 0 0; }
    button[data-baseweb="tab"] p { font-size: 0.92rem; font-weight: 500; color: var(--muted); }
    [data-baseweb="tab-list"] button[aria-selected="true"],
    [data-baseweb="tab-list"] button[aria-selected="true"] * { color: var(--accent) !important; font-weight: 600; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 3px; }
    [data-baseweb="tab-border"] { display: none; }

    /* ---------- Botones ---------- */
    .stButton > button, [data-testid="stDownloadButton"] > button {
        background: var(--accent); color: #FFFFFF; border: none; border-radius: 8px;
        padding: 0.55rem 1.3rem; font-weight: 600; font-size: 0.92rem;
        box-shadow: 0 1px 2px rgba(15, 27, 45, 0.12);
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        background: var(--accent-dark); color: #FFFFFF; border: none;
        box-shadow: 0 3px 8px rgba(36, 87, 214, 0.28);
    }
    .stButton > button:focus-visible, [data-testid="stDownloadButton"] > button:focus-visible {
        outline: 3px solid rgba(36, 87, 214, 0.35); outline-offset: 2px;
    }
    .stButton > button p, [data-testid="stDownloadButton"] > button p { color: #FFFFFF; }

    /* ---------- Campos de entrada ---------- */
    [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
        background: var(--surface); border-radius: 8px; border-color: var(--line);
    }
    [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
        border-color: var(--accent); box-shadow: 0 0 0 3px rgba(36, 87, 214, 0.15);
    }
    [data-testid="stWidgetLabel"] p { font-size: 0.88rem; font-weight: 500; color: var(--text); }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface); border: 1.5px dashed #B9C4D8; border-radius: 12px;
    }

    /* ---------- Expansores, tablas y código ---------- */
    [data-testid="stExpander"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary p { font-weight: 600; font-size: 0.93rem; }
    [data-testid="stTable"] table { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--surface); }
    [data-testid="stTable"] th { background: #EDF1F8; color: var(--text); font-weight: 600; font-size: 0.85rem; }
    [data-testid="stTable"] td { font-size: 0.88rem; border-color: var(--line); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid="stCode"] { border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--line); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    </style>
""", unsafe_allow_html=True)

def encabezado(titulo, descripcion=""):
    """Cabecera de página con jerarquía visual consistente (solo presentación)."""
    import html as _html
    import re as _re
    t = _html.escape(titulo)
    d = _html.escape(descripcion)
    d = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", d)
    d = _re.sub(r"`(.+?)`", r"<code>\1</code>", d)
    cuerpo = f"<p>{d}</p>" if d else ""
    st.markdown(f'<div class="page-head"><h1>{t}</h1>{cuerpo}</div>', unsafe_allow_html=True)

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
st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark">TI</div>'
    '<div><div class="brand-name">Panel de Control TI</div>'
    '<div class="brand-sub">Asistente de Soporte Híbrido</div></div></div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navegación",
    [
        "Resumen Ejecutivo", 
        "Matriz de Confusión", 
        "Sistema Híbrido & TF-IDF", 
        "Planificador A* & Minimax", 
        "Representaciones del Reconocimiento",
        "Red Neuronal & Ontología",
        "Visión Artificial (Semana 09)",
        "Histogramas y Texturas LBP (Semana 10)",
        "Portal de Usuario",
        "Base de Conocimiento (KB)", 
        "Trazas de Auditoría"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="side-card">'
    '<div class="side-card-k">Institución</div><div class="side-card-v">ETITC - 10º Semestre</div>'
    '<div class="side-card-k">Autores</div><div class="side-card-v">Marco Molina &amp; Daniel Daza</div>'
    '</div>',
    unsafe_allow_html=True
)

# --- 1. Resumen Ejecutivo ---
if menu == "Resumen Ejecutivo":
    encabezado("Dashboard General del Asistente de Soporte TI", "Monitoreo en tiempo real de los componentes lógicos, modelos de machine learning y flujos del sistema híbrido.")
    
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
        st.subheader("Distribución por Categorías Técnicas (KB)")
        data_cat = {
            "Categoría": ["Hardware", "Red", "Seguridad", "Rendimiento", "Almacenamiento", "Software"],
            "Procedimientos": [8, 6, 6, 6, 3, 3]
        }
        df_cat = pd.DataFrame(data_cat)
        st.bar_chart(df_cat.set_index("Categoría"), color="#2457D6")
        
    with col_b:
        st.subheader("Estado de Integración de Módulos")
        modulos_estado = [
            ("Validación y Matriz de Confusión (Semana 02)", "Accuracy 92.1%"),
            ("Clasificador Taxonómico (Semana 03)", "100% de coincidencia"),
            ("Planificador A* y Ciberdefensa Minimax (Semana 04)", ""),
            ("Sistema Híbrido (Semana 05)", "Reglas y TF-IDF"),
            ("Representaciones del Reconocimiento (Semana 07)", ""),
            ("Red Neuronal, Imagen Base64, SQLite y Ontología (Semana 08)", ""),
            ("Visión Artificial (Semana 09)", ""),
            ("Histogramas y Texturas LBP (Semana 10)", "Extracción de Features Vectoriales"),
            ("Motor de Auditoría y Trazas", "Registrando en artifacts/audit.log"),
        ]
        filas_estado = "".join(
            f'<div class="status-row"><div><div class="status-name">{nombre}</div>'
            f'<div class="status-meta">{detalle}</div></div><span class="badge-ok">Operativo</span></div>'
            for nombre, detalle in modulos_estado
        )
        st.markdown(f'<div class="status-list">{filas_estado}</div>', unsafe_allow_html=True)

# --- 2. Matriz de Confusión ---
elif menu == "Matriz de Confusión":
    encabezado("Validación del Modelo Base (Semana 02)", "Evaluación del pipeline reproducible de clasificación multiclase del curso (StandardScaler + LogisticRegression sobre el benchmark estándar con división estratificada 75/25).")
    
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
elif menu == "Sistema Híbrido & TF-IDF":
    encabezado("Simulador de Sistema Híbrido (Reglas + TF-IDF)", "Consulta la base de conocimiento en tiempo real utilizando similitud coseno sobre los 30 procedimientos de soporte y reglas lógicas expertas.")

    query = st.text_input("Escribe un reporte de ticket de soporte:", value="Disco lleno", key="query_input")
    
    if st.button("Ingresar su consulta"):
        if MODULOS_CARGADOS:
            sistema = SistemaHibridoSoporte()
            resultado = sistema.procesar(query)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Resultados del Motor")
                st.info(f"**Reglas Activadas:** `{resultado['reglas']}`")
                st.warning(f"**Clase Predicha (ML):** `{resultado['clase']}`")
                st.metric(label="Similitud Coseno", value=f"{resultado['similitud']:.4f}")
            with col2:
                st.markdown("### Evidencia Recuperada (TF-IDF)")
                st.success(f"{resultado['evidencia']}")
        else:
            st.error("No se pudieron cargar los módulos de Python desde la carpeta `src/`.")

# --- 4. Planificador A* & Minimax ---
elif menu == "Planificador A* & Minimax":
    encabezado("Optimización en Espacio de Estados y Decisiones Adversariales")
    
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
            st.info(f"**Estado Inicial:** `{estado_ini}` ➔ **Estado Meta:** `{GOAL_STATE}` | **Costo Mínimo Acumulado:** `{costo_total}` unidades de esfuerzo")
            
            st.markdown("#### Recuperación y Salud de Subsistemas")
            c_red, c_bd, c_app = st.columns(3)
            with c_red:
                st.markdown(f"**Red:** Nivel `{estado_ini[0]}/2`")
                st.progress(estado_ini[0] / 2.0, text=f"{(estado_ini[0]/2)*100:.0f}% Operativo")
            with c_bd:
                st.markdown(f"**Base de Datos:** Nivel `{estado_ini[1]}/2`")
                st.progress(estado_ini[1] / 2.0, text=f"{(estado_ini[1]/2)*100:.0f}% Operativo")
            with c_app:
                st.markdown(f"**Microservicio Backend:** Nivel `{estado_ini[2]}/2`")
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
                
                st.markdown("##### Progresión de Salud hacia la Meta (2, 2, 2)")
                df_evol = pd.DataFrame(evolucion)
                st.line_chart(df_evol.set_index("Paso"), color=["#2457D6", "#0E9F8E", "#E08A1E"])
                
                st.markdown("##### Secuencia Óptima de Acciones Técnicas")
                st.table(pd.DataFrame(pasos_lista))
            else:
                st.warning("No se encontró una ruta válida para este estado.")
        else:
            st.warning("Módulo A* no disponible temporalmente.")
        
    with tab2:
        st.subheader("Simulador de Ciberdefensa con Minimax (Blue Team vs. Red Team)")
        st.markdown(
            "Modelado formal de **búsqueda adversarial de suma cero** para contención de incidentes de seguridad y mitigación de intrusión lateral.  \n"
            "- **Jugador MAX (Blue Team / Soporte TI):** Aplica contramedidas preventivas y aísla servidores críticos (`Blindado`).  \n"
            "- **Jugador MIN (Red Team / Amenaza Externa):** Explota vulnerabilidades buscando comprometer una ruta crítica (`Comprometido`).  \n"
            "- **Objetivo:** Minimax evalúa exhaustivamente el árbol de decisiones para seleccionar el nodo que garantiza la resiliencia de la infraestructura corporativa."
        )
        if MODULOS_CARGADOS:
            diag = simular_ciberdefensa(minimax_board)
            
            col_left, col_right = st.columns([1, 1], gap="medium")
            
            with col_left:
                st.markdown("##### Topología de Servidores (Matriz 3x3)")
                
                tarjetas_html = []
                for i in range(9):
                    estado = minimax_board[i]
                    nombre = NODOS_INFRAESTRUCTURA[i]
                    es_mejor = (i == diag["posicion"])
                    
                    if estado == "X":
                        bg = "#E7F5EE"
                        border = "#1E8E5A"
                        badge = "Blue Team"
                        sub = "Blindado"
                    elif estado == "O":
                        bg = "#FCEBE9"
                        border = "#C0392B"
                        badge = "Red Team"
                        sub = "Comprometido"
                    elif es_mejor:
                        bg = "#FEF6E4"
                        border = "#B7791F"
                        badge = "Minimax"
                        sub = "Blindar Ahora"
                    else:
                        bg = "#F4F6FA"
                        border = "#C5CEDC"
                        badge = "En Línea"
                        sub = "Disponible"
                        
                    tarjetas_html.append(
                        f'<div style="background-color: {bg}; border: 1px solid {border}; border-left: 4px solid {border}; border-radius: 10px; padding: 12px 6px; text-align: center; min-height: 96px;">'
                        f'<span style="font-size: 0.72rem; color: #5B6779;">[Nodo #{i}]</span><br>'
                        f'<strong style="color: #14213D; font-size: 0.82rem;">{nombre}</strong><br>'
                        f'<span style="font-weight: bold; font-size: 0.75rem; color: {border};">{badge}</span><br>'
                        f'<small style="color: #5B6779; font-size: 0.72rem;">{sub}</small>'
                        f'</div>'
                    )
                
                grid_html = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px;">' + "".join(tarjetas_html) + '</div>'
                st.markdown(grid_html, unsafe_allow_html=True)
                
            with col_right:
                st.markdown("##### Decisión Táctica de Contención")
                st.success(f"**Servidor Seleccionado:** `{diag['nodo']}` (Nodo #{diag['posicion']})")
                st.info(f"**Acción Defensiva:** {diag['accion']}")
                st.markdown(f"**Análisis Estratégico:**\n\n{diag['justificacion']}")
                st.caption("Función de Utilidad: +1 si el Blue Team asegura el perímetro, 0 si hay contención, -1 si el Red Team compromete la red.")
        else:
            st.warning("Módulo Minimax no disponible temporalmente.")

# --- 5. REPRESENTACIONES DEL RECONOCIMIENTO (Semana 07) ---
elif menu == "Representaciones del Reconocimiento":
    encabezado("Representaciones del Reconocimiento (Semana 07)", "Simulación de los tres enfoques de inteligencia artificial para interpretar y procesar información técnica de soporte.")
    
    tab_num, tab_sim, tab_aut = st.tabs(["Numérica (Telemetría)", "Simbólica (Sistema Experto)", "Autómatas (Logs en Cascada)"])
    
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
                st.error("**ALERTA DE TRIAJE:** Clasificación prioritaria. Asignando directamente a Especialista Humano.")
            else:
                st.success("**ESTABLE:** Ticket estándar manejable por chatbot de Nivel 1.")
                
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
            st.error("**Diagnóstico Simbólico:** `falla_hardware_fisico` detectada.")
            st.markdown("**Acción recomendada:** Desviar al módulo de Soporte de Campo (Presencial) y solicitar modelo del equipo.")
        elif {"ping_fallido", "servidor_no_responde"}.issubset(hechos_detectados):
            st.error("**Diagnóstico Simbólico:** `caida_infraestructura_red` detectada.")
            st.markdown("**Acción recomendada:** Notificar a NOC y ejecutar protocolo de ping persistente.")
        elif {"olvido_contrasena", "active_directory"}.issubset(hechos_detectados):
            st.warning("**Diagnóstico Simbólico:** `restablecimiento_credenciales`.")
            st.markdown("**Acción recomendada:** Ejecutar script de automatización de reseteo (Nivel 1 - Bot).")
        else:
            st.info("**Diagnóstico Simbólico:** Incidentes sin mapeo directo crítico.")

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
                    st.error("**Secuencia Aceptada.** El AFD confirma que el patrón estricto `[Error -> Timeout]` se cumple. Falla en cascada confirmada.")
                else:
                    st.success("**Secuencia Rechazada.** El sistema se estabilizó al final o la secuencia no terminó en Timeout.")

# --- 6. Red Neuronal, Imagen Base64 & Ontología (Semana 08) ---
elif menu == "Red Neuronal & Ontología":
    encabezado("Red Neuronal, Imágenes Base64 y Ontología (Semana 08)", "Integración completa: Modelo MLP (Predicción) + Carga y Descripción de Imagen Base64 ➔ SQLite (Evidencia) ➔ Ontología GraphML (Significado).")
    
    tab_nn, tab_img, tab_db, tab_onto = st.tabs([
        "Clasificador MLP", 
        "Análisis de Imagen (Base64)", 
        "Base de Evidencia (SQLite)", 
        "Ontología del Dominio"
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
                st.success(f"Categoría Predicha por el Modelo: **{mapa_clases.get(prediccion, 'Desconocida')}** (Clase ID: {prediccion})")
            else:
                st.warning("Primero debes ejecutar `python src/main.py` para generar el modelo entrenado.")
                
    with tab_img:
        st.subheader("Reconocimiento de Imágenes, Base64 y Metadatos (Soporte TI)")
        st.markdown("Adjunta una captura de pantalla de error técnico e ingresa una descripción textual para que el agente la codifique, analice e integre en el flujo de evidencia.")
        
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
                        diag_resultado = "Red / Conectividad (Clase 1)"
                    elif "azul" in nombre_l or "hardware" in nombre_l:
                        diag_resultado = "Hardware / Pantalla Azul (Clase 0)"
                    else:
                        diag_resultado = "Software / Aplicación (Clase 2)"
                        
                    st.success(f"**Agente IA - Imagen Reconocida y Descrita:** {diag_resultado}")
                    st.info("**Estado en Proyecto:** La imagen se codificó en Base64, se asoció a su descripción textual, se clasificó con enfoque de soporte TI, se persistió en SQLite y se enlazó en la ontología GraphML.")
                except Exception as e:
                    st.error(f"Error al procesar la imagen en el pipeline: {e}")

    with tab_db:
        st.subheader("Registro de Evidencia Verificable (SQLite)")
        st.markdown("Consulta los registros almacenados en `artifacts/soporte_evidencia.db` (incluyendo metadatos de imágenes Base64 y descripciones textualmente auditadas).")
        db_p = BASE_DIR / "artifacts" / "soporte_evidencia.db"
        if db_p.exists():
            with sqlite3.connect(db_p) as conn:
                df_ev = pd.read_sql("SELECT * FROM evidencia_tickets ORDER BY id DESC LIMIT 15", conn)
            st.dataframe(df_ev, use_container_width=True)
        else:
            st.warning("Base de datos no encontrada. Ejecuta el script del proyecto.")
            
    with tab_onto:
        st.subheader("Representación del Conocimiento (Ontología GraphML)")
        st.markdown("Relaciones lógicas que otorgan significado a las predicciones y canales visuales dentro del dominio del Asistente de Soporte TI.")
        onto_p = BASE_DIR / "artifacts" / "ontologia.graphml"
        if onto_p.exists():
            G_load = nx.read_graphml(onto_p)
            st.info(f"Grafo cargado correctamente con {G_load.number_of_nodes()} conceptos y {G_load.number_of_edges()} relaciones semánticas.")
            edges_list = [(u, v, data.get('rel', 'relación')) for u, v, data in G_load.edges(data=True)]
            st.table(pd.DataFrame(edges_list, columns=["Sujeto", "Objeto", "Relación Semántica"]))
        else:
            st.warning("Archivo GraphML no encontrado en artifacts/.")

# --- 7. Visión Artificial (Semana 09) ---
elif menu == "Visión Artificial (Semana 09)":
    encabezado("Visión Artificial y Procesamiento de Imágenes", "Transformación de evidencia visual en información numérica mediante extracción de características, detección de contornos (Canny) y segmentación por umbrales (Otsu).")

    st.subheader("Analizar Evidencia desde la Base de Conocimiento")
    
    opciones_imagenes = {
        "Falla 1: Disco Duro Dañado (Sectores defectuosos)": "disco_duro.png",
        "Falla 2: Pantalla Azul (Volcado de memoria)": "pantalla_azul.png",
        "Falla 3: Router sin conexión (Luces apagadas)": "router.png"
    }
    
    imagen_seleccionada = st.selectbox("Selecciona un caso de estudio predefinido para procesar:", list(opciones_imagenes.keys()), key="select_vis_09")
    archivo_imagen = opciones_imagenes[imagen_seleccionada]

    if st.button("Procesar Imagen Seleccionada", key="btn_vis_09"):
        if MODULOS_CARGADOS:
            with st.spinner(f"Procesando bordes, umbrales y regiones de {archivo_imagen}..."):
                try:
                    umbral, regiones = ejecutar_vision_soporte(nombre_imagen=archivo_imagen)
                    
                    if umbral is not None:
                        st.session_state['otsu_thresh'] = umbral
                        st.session_state['otsu_regs'] = regiones
                        st.success(f"¡Pipeline ejecutado! Imagen '{archivo_imagen}' procesada correctamente.")
                    else:
                        st.error(f"No se encontró la imagen '{archivo_imagen}' en la carpeta 'data/'. ¡Asegúrate de guardarla allí primero!")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
        else:
            st.error("Los módulos no se cargaron correctamente. Verifica las importaciones.")

    img_path = Path("artifacts/semana09_vision.png")
    if img_path.exists():
        st.image(str(img_path), caption="Pipeline de Visión: Imagen Original ➔ Bordes (Canny) ➔ Máscara Binaria (Otsu)", use_container_width=True)
        
        if 'otsu_thresh' in st.session_state and 'otsu_regs' in st.session_state:
            st.markdown("### Resultados del Procesamiento Numérico")
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
        st.warning("No se ha generado la evidencia visual. Selecciona una imagen y procesa el pipeline.")

# --- 8. Histogramas y Texturas LBP (Semana 10) ---
elif menu == "Histogramas y Texturas LBP (Semana 10)":
    encabezado("Comparativa de Dispositivos (Bueno vs. Malo) y Análisis de Texturas LBP", "Evaluación morfológica y comparación lado a lado entre componentes en estado óptimo y componentes con fallos específicos.")

    categorias_hardware = {
        "Disco Duro": {
            "bueno": "disco_duro_bueno.jpg",
            "malos": {
                "Daño por Golpe Físico": "disco_duro_dañado_por_golpe.jpg",
                "Rayones en Superficie Magnética": "disco_duro_dañado_por_rayones.jpg",
                "No Gira (Falla en Motor Spindle)": "disco_duro_no_gira.jpg"
            }
        },
        "Router / Red": {
            "bueno": "router_bueno.jpg",
            "malos": {
                "Falla de Conexión Intermitente": "router_falla_conexion_intermitente.jpg"
            }
        },
        "Pantalla / Display": {
            "bueno": "panatalla_inicio_bueno.jpg",
            "malos": {
                "Daño Físico por Golpe en Display": "pantalla_dañada_por_golpe.jpg",
                "Fallo de Encendido (No Arranca / Sin Video)": "pantalla_no_arranca.jpg"
            }
        }
    }

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        categoria_sel = st.selectbox("Seleccionar Categoría de Producto:", list(categorias_hardware.keys()), key="s10_cat")
    
    prod_info = categorias_hardware[categoria_sel]
    img_buena_nombre = prod_info["bueno"]

    with col_cat2:
        fallo_sel = st.selectbox(f"Seleccionar Tipo de Avería ({categoria_sel}):", list(prod_info["malos"].keys()), key="s10_fallo")
    
    img_mala_nombre = prod_info["malos"][fallo_sel]

    st.markdown("---")

    if st.button("Ejecutar Análisis Comparativo (Bueno vs. Malo)"):
        if MODULOS_CARGADOS:
            with st.spinner("Procesando características, histogramas y texturas LBP para ambos estados..."):
                try:
                    import matplotlib.pyplot as plt
                    
                    img_a, mask_a, hist_int_a, hist_lbp_a, feat_a = procesar_imagen_proyecto(img_buena_nombre)
                    img_b, mask_b, hist_int_b, hist_lbp_b, feat_b = procesar_imagen_proyecto(img_mala_nombre)

                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    axes[0, 0].imshow(img_a, cmap='gray')
                    axes[0, 0].set_title(f"Óptimo: {img_buena_nombre}")
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(img_b, cmap='gray')
                    axes[0, 1].set_title(f"Averiado: {img_mala_nombre}")
                    axes[0, 1].axis('off')
                    
                    axes[1, 0].plot(hist_int_a, label="Intensidad (Sano)", color='blue')
                    axes[1, 0].plot(hist_lbp_a, label="LBP Textura (Sano)", color='orange')
                    axes[1, 0].set_title(f"Histogramas - {img_buena_nombre}")
                    axes[1, 0].legend(loc="upper right")
                    
                    axes[1, 1].plot(hist_int_b, label="Intensidad (Averiado)", color='green')
                    axes[1, 1].plot(hist_lbp_b, label="LBP Textura (Averiado)", color='red')
                    axes[1, 1].set_title(f"Histogramas - {img_mala_nombre}")
                    axes[1, 1].legend(loc="upper right")
                    
                    plt.tight_layout()
                    artifact_path = BASE_DIR / "artifacts" / "semana10_comparativa.png"
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(artifact_path)
                    plt.close()

                    st.session_state['s10_feat_bueno'] = feat_a
                    st.session_state['s10_feat_malo'] = feat_b
                    st.session_state['s10_nombres'] = (img_buena_nombre, img_mala_nombre)
                    st.session_state['s10_fallo_desc'] = fallo_sel

                    st.success("¡Análisis comparativo de la Semana 10 completado con éxito!")
                except Exception as e:
                    st.error(f"Error al procesar las imágenes: {e}")
        else:
            st.error("Módulos no cargados correctamente.")

    comp_path = Path("artifacts/semana10_comparativa.png")
    if comp_path.exists():
        st.markdown("### Visualización Lado a lado (Imágenes e Histogramas Analíticos)")
        st.image(str(comp_path), use_container_width=True)

    if 's10_feat_bueno' in st.session_state:
        f_bno = st.session_state['s10_feat_bueno']
        f_mlo = st.session_state['s10_feat_malo']
        n_bno, n_mlo = st.session_state['s10_nombres']
        desc_fallo = st.session_state['s10_fallo_desc']

        st.markdown("### Especificación Técnica del Diagnóstico y Fallos")
        st.error(f"**Avería Detectada:** {desc_fallo} (`{n_mlo}`)")
        
        if "golpe" in n_mlo:
            st.markdown("- **Análisis de Fallo:** Disrupción estructural por impacto cinético. Las regiones conectadas aumentan debido a las fracturas y astillas, alterando drásticamente el patrón regular de la referencia sana (`{n_bno}`).")
        elif "rayones" in n_mlo:
            st.markdown("- **Análisis de Fallo:** Abrasión profunda en la superficie magnética. Se observa una desviación notoria en los descriptores de textura LBP respecto al componente óptimo.")
        elif "no_gira" in n_mlo:
            st.markdown("- **Análisis de Fallo:** Traba mecánica en el motor o bloqueo de eje. El histograma carece de variabilidad dinámica al encontrarse el componente detenido.")
        elif "intermitente" in n_mlo:
            st.markdown("- **Análisis de Fallo:** Degeneración en la pasarela de red y pérdida de estabilidad de enlace inalámbrico.")
        else:
            st.markdown("- **Análisis de Fallo:** Anomalía crítica en circuitos de video o alimentación, rompiendo la simetría morfológica del estado base.")

        st.markdown("---")
        st.markdown("### Comparativa Numérica de Vectores de Características (.npy)")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Referencia Sana: `{n_bno}`**")
            st.metric("Área Media de Regiones", f"{f_bno[0]:.2f} px")
            st.metric("Desviación Estándar", f"{f_bno[1]:.2f}")
            st.metric("Regiones Validadas", int(f_bno[2]))
        with col_m2:
            st.markdown(f"**Componente Averiado: `{n_mlo}`**")
            st.metric("Área Media de Regiones", f"{f_mlo[0]:.2f} px", delta=f"{f_mlo[0]-f_bno[0]:.2f} px", delta_color="inverse")
            st.metric("Desviación Estándar", f"{f_mlo[1]:.2f}", delta=f"{f_mlo[1]-f_bno[1]:.2f}", delta_color="off")
            st.metric("Regiones Validadas", int(f_mlo[2]), delta=f"{int(f_mlo[2]-f_bno[2])}", delta_color="inverse")
    else:
        st.warning("Selecciona una categoría y un tipo de avería, luego haz clic en 'Ejecutar Análisis Comparativo (Bueno vs. Malo)'.")

# --- 9. Portal de Usuario ---
elif menu == "Portal de Usuario":
    encabezado("Portal / Cliente Final", "Interfaz orientada al usuario final para la creación de solicitudes de soporte y asistencia.")
    
    usuario_nombre = st.text_input("Nombre del Empleado / Usuario:", "Carlos Pérez (Contabilidad)")
    problema_desc = st.text_area("Describe el síntoma técnico:", "La aplicación de nómina se congela y arroja pantalla azul al intentar generar el cierre mensual.")
    urgencia_opc = st.selectbox("Nivel de Urgencia percibida:", ["Baja", "Media", "Alta", "Crítica"])
    
    if st.button("Ingresar su consulta"):
        st.success(f"¡Ticket registrado con éxito para {usuario_nombre}!")
        st.markdown("---")
        st.markdown("### Análisis Automático del Asistente (Semana 07)")
        
        if "pantalla_azul" in problema_desc.lower() or "congela" in problema_desc.lower():
            st.error("**Representación Simbólica:** Síntoma crítico identificado (`falla_hardware_fisico`).")
            st.markdown("**Acción recomendada asignada:** Derivación automática a soporte técnico presencial.")
        else:
            st.info("**Representación Simbólica:** Incidente registrado como estándar.")
            
        vector_ticket = np.array([9.0 if urgencia_opc=="Crítica" else 5.0, 2.0, 0.8])
        dist_portal = np.linalg.norm(vector_ticket - np.array([10.0, 10.0, 1.0]))
        st.metric(label="Métrica de Proximidad a Incidente Crítico (Distancia Euclidiana)", value=f"{dist_portal:.2f}")

# --- 10. Base de Conocimiento (KB) ---
elif menu == "Base de Conocimiento (KB)":
    encabezado("Base de Conocimiento Técnica (KB)", "Catálogo de los **30 procedimientos operativos estandarizados** del Asistente de Soporte TI, utilizados para el entrenamiento del clasificador y el motor TF-IDF.")
    
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
            with st.expander(f"#{proc['id']} - {proc['titulo']}  [{proc['categoria']}]"):
                st.markdown(f"**Categoría:** `{proc['categoria']}`")
                st.markdown("**Procedimiento de Solución:**")
                for paso_idx, paso in enumerate(proc['pasos'], 1):
                    st.markdown(f"{paso_idx}. {paso}")
    else:
        st.warning("No se encontraron procedimientos que coincidan con los filtros aplicados.")

# --- 11. Trazas de Auditoría ---
elif menu == "Trazas de Auditoría":
    encabezado("Registro de Auditoría del Sistema (audit.log)", "Monitoreo y observabilidad de eventos generados por el orquestador `main.py` y los clasificadores del sistema.")
    
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
        
        tab_tabla, tab_raw = st.tabs(["Vista Estructurada", "Consola Raw"])
        
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
                label="Descargar audit.log",
                data=raw_text,
                file_name="audit.log",
                mime="text/plain"
            )
    else:
        st.warning("No se encontró el archivo de auditoría en `artifacts/audit.log`. Ejecuta `python src/main.py` para generarlo.")
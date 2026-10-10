# src/classifiers/reconocimiento.py
from pathlib import Path
import pickle
import sqlite3
import networkx as nx
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Definición de la ruta raíz del proyecto y la carpeta de artefactos
ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True) # Crea la carpeta si no existe
KB_PATH = ROOT / "data" / "base_conocimiento.txt" # Ruta a la base de conocimiento en texto

def buscar_diagnostico_en_kb(texto_analisis):
    """Lee directamente el archivo físico de la base de conocimiento para extraer el procedimiento real."""
    # Verifica si el archivo de la base de conocimiento existe
    if not KB_PATH.exists():
        return "Procedimiento estándar de soporte (Archivo KB no encontrado)"
    
    texto_analisis = texto_analisis.lower() # Pasa el texto a minúsculas para estandarizar la búsqueda
    mejor_coincidencia = "Clase Genérica: Procedimiento Estándar de Soporte"
    
    # Abre y lee línea por línea el archivo de la base de conocimiento
    with open(KB_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or "->" not in linea: # Ignora líneas vacías o mal formateadas
                continue
            
            problema_kb = linea.split("->")[0].lower() # Extrae el problema o síntoma de la KB
            palabras = [p for p in problema_kb.split() if len(p) > 3] # Filtra palabras con más de 3 letras
            # Busca coincidencias de palabras clave evitando palabras comunes de relleno
            if any(palabra in texto_analisis for palabra in palabras if palabra not in ["revisar", "verificar"]):
                mejor_coincidencia = f"Match KB Directo: {linea}"
                break
                
    return mejor_coincidencia

def ejecutar_semana_08(imagen_bytes=None, nombre_archivo="", descripcion_texto=""):
    print("=== INICIANDO PIPELINE SEMANA 08: RECONOCIMIENTO CON LECTURA DE KB ===")
    
    # 1. Simulación de datos de telemetría
    np.random.seed(42) # Fija la semilla aleatoria para reproducibilidad
    X = np.random.rand(200, 4) * 10 # Genera 200 muestras sintéticas con 4 características numéricas
    y = np.random.randint(0, 4, size=200) # Genera etiquetas aleatorias entre 0 y 3 (4 clases de soporte TI)

    # Divide los datos en conjuntos de entrenamiento (75%) y prueba (25%) de forma estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Entrenamiento del Modelo MLP y cálculo de Accuracy
    # Configura una Red Neuronal Perceptrón Multicapa con dos capas ocultas (32 y 16 neuronas)
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(X_train, y_train) # Entrena el modelo con los datos de entrenamiento
    pred = model.predict(X_test) # Realiza predicciones sobre el conjunto de prueba
    acc = accuracy_score(y_test, pred) # Calcula la precisión global (accuracy)
    print(f"Accuracy del Modelo MLP (Soporte TI): {round(acc, 4)}")

    # Serializa y guarda el modelo entrenado en un archivo binario .pkl usando pickle
    with (ARTIFACTS / "modelo_mlp.pkl").open("wb") as file:
        pickle.dump(model, file)

    # 3. Procesamiento y Búsqueda Directa en la Base de Conocimiento
    imagen_b64 = ""
    prediccion_imagen = "No adjunta"
    
    # Si se reciben bytes de imagen, se codifican en Base64 y se analiza la KB
    if imagen_bytes:
        imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")
        if not descripcion_texto:
            descripcion_texto = f"Evidencia visual: {nombre_archivo}"
        
        texto_busqueda = f"{nombre_archivo} {descripcion_texto}"
        prediccion_imagen = buscar_diagnostico_en_kb(texto_busqueda)
    else: # Si no hay imagen, se asignan valores de simulación por defecto
        imagen_b64 = base64.b64encode(b"dummy_image").decode("utf-8")
        descripcion_texto = "Imagen de simulación por defecto."
        prediccion_imagen = "Imagen simulada (Estándar)"

    print(f"Resultado de la KB: {prediccion_imagen}")

    # 4. Registro Acumulativo en SQLite (Un solo registro limpio por ejecución)
    db_path = ARTIFACTS / "soporte_evidencia.db"
    with sqlite3.connect(db_path) as con:
        # Crea la tabla SQLite si no existe previamente con los campos requeridos
        con.execute(
            "CREATE TABLE IF NOT EXISTS evidencia_tickets("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "vector_entrada TEXT, "
            "imagen_base64 TEXT, "
            "descripcion_imagen TEXT, "
            "diagnostico_imagen TEXT, "
            "etiqueta_real INTEGER, "
            "prediccion_ia INTEGER, "
            "fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        
        # Prepara la tupla con la evidencia actual de la ejecución
        registro = (
            str(X_test[0].tolist()), 
            imagen_b64, 
            descripcion_texto, 
            prediccion_imagen, 
            int(y_test[0]), 
            int(pred[0])
        )
        
        # Inserta el registro de manera segura en la base de datos
        con.execute(
            "INSERT INTO evidencia_tickets(vector_entrada, imagen_base64, descripcion_imagen, diagnostico_imagen, etiqueta_real, prediccion_ia) VALUES (?, ?, ?, ?, ?, ?)",
            registro
        )
        con.commit() # Confirma los cambios en la base de datos

    # 5. Construcción de Ontología con 6 conceptos y 7 relaciones semánticas formales
    G = nx.DiGraph() # Inicializa un grafo dirigido utilizando NetworkX
    # Añade aristas (conexiones) que representan relaciones semánticas conceptuales
    G.add_edge("modelo_mlp", "ticket_soporte", rel="reconoce patrones en")
    G.add_edge("ticket_soporte", "categoria_falla", rel="representa problema de")
    G.add_edge("modelo_mlp", "prediccion_ia", rel="produce salida de")
    G.add_edge("prediccion_ia", "categoria_falla", rel="asigna clase a")
    G.add_edge("base_conocimiento_txt", "categoria_falla", rel="explica procedimiento de")
    G.add_edge("imagen_diagnostico_ti", "ticket_soporte", rel="evidencia incidente en")
    G.add_edge("base_conocimiento_txt", "imagen_diagnostico_ti", rel="contextualiza analisis de")

    ontologia_path = ARTIFACTS / "ontologia.graphml"
    nx.write_graphml(G, ontologia_path) # Exporta el grafo en formato GraphML estándar
    print(f"Ontología GraphML generada con {G.number_of_nodes()} conceptos y {G.number_of_edges()} relaciones en: {ontologia_path}")
    print("=== PIPELINE COMPLETO CON LECTURA DE KB Y ONTOLOGÍA VALIDADA ===")

if __name__ == "__main__":
    ejecutar_semana_08()
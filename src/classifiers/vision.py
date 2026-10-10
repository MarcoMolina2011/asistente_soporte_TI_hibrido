# src/classifiers/vision.py
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Configura el backend de Matplotlib en modo no interactivo para evitar errores de interfaz gráfica en entornos web o CLI
import matplotlib.pyplot as plt
# Importa submódulos específicos de scikit-image para entrada/salida (io), conversión de color (color), detección de bordes (feature), umbrales (filters) y etiquetado (measure)
from skimage import io, color, feature, filters, measure
import numpy as np

# Configuración robusta de rutas absolutas (subiendo 3 niveles desde src/classifiers hasta la raíz del proyecto)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" # Directorio donde se almacenan las imágenes de entrada
ARTIFACTS_DIR = ROOT / "artifacts" # Directorio para guardar resultados y evidencias generadas
ARTIFACTS_DIR.mkdir(exist_ok=True) # Crea la carpeta de artefactos si no existe previamente

# Define la función principal del pipeline, aceptando por defecto el archivo "disco_duro.png"
def ejecutar_vision_soporte(nombre_imagen="disco_duro.png"):
    print(f"=== INICIANDO PIPELINE DE VISIÓN: SEMANA 09 ({nombre_imagen}) ===")
    
    # Construye la ruta completa hacia la imagen seleccionada dentro de la carpeta data
    img_path = DATA_DIR / nombre_imagen
    if not img_path.exists(): # Valida si la imagen existe físicamente
        print(f"Error: No se encontró la imagen en {img_path}")
        return None, None # Retorna nulos si ocurre un error de archivo no encontrado

    # Carga la imagen desde el disco usando scikit-image
    image_color = io.imread(img_path)
    # Comprueba si la imagen tiene canales de color (RGB/RGBA, shape de 3 dimensiones)
    if len(image_color.shape) == 3:
        image = color.rgb2gray(image_color) # Convierte la imagen a escala de grises si es a color
    else:
        image = image_color # La mantiene igual si ya está en escala de grises

    sigma_val = 2.0 # Define el valor de sigma para el filtro gaussiano de suavizado en Canny
    edges = feature.canny(image, sigma=sigma_val) # Aplica el algoritmo de detección de contornos Canny

    threshold = filters.threshold_otsu(image) # Calcula automáticamente el umbral óptimo global mediante el método de Otsu
    mask = image > threshold # Genera una máscara binaria binarizando la imagen con base en el umbral de Otsu

    labels = measure.label(mask) # Etiqueta los grupos de píxeles conectados en regiones distintas
    num_regiones = labels.max() # Obtiene el número total de regiones conectadas encontradas

    print(f"Umbral automático Otsu obtenido: {round(threshold, 4)}")
    print(f"Número de regiones conectadas encontradas: {num_regiones}")

    # Configura una figura de Matplotlib con una cuadrícula de 1 fila y 3 columnas para mostrar la comparativa visual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray") # Muestra la imagen original en escala de grises
    axes[0].set_title("Original (Escala de Grises)")
    axes[1].imshow(edges, cmap="gray") # Muestra el mapa de contornos detectados por Canny
    axes[1].set_title(f"Contornos (Canny, sigma={sigma_val})")
    axes[2].imshow(mask, cmap="gray") # Muestra la máscara binaria obtenida por el umbral de Otsu
    axes[2].set_title("Máscara Binaria (Otsu)")

    for ax in axes:
        ax.axis("off") # Desactiva los ejes numéricos en los tres subgráficos para una presentación más limpia

    fig.tight_layout() # Ajusta automáticamente los espacios entre gráficos
    output_path = ARTIFACTS_DIR / "semana09_vision.png" # Define la ruta de salida del artefacto visual combinado
    fig.savefig(output_path, dpi=160) # Guarda la figura generada en el disco con alta resolución (160 DPI)
    print(f"Evidencia visual guardada exitosamente en: {output_path}")
    print("=== PIPELINE COMPLETADO ===")
    
    # Retorna los valores numéricos calculados (umbral de Otsu y número de regiones) para usarlos en el dashboard
    return threshold, num_regiones

if __name__ == "__main__":
    ejecutar_vision_soporte() # Ejecuta la función por defecto si el script se invoca directamente desde la terminal
# src/classifiers/texturas.py

# Importación de la clase Path para el manejo de rutas de archivos de forma independiente del sistema operativo
from pathlib import Path
# Importación del módulo matplotlib para la gestión de configuraciones de ploteo
import matplotlib
# Configuración del backend en 'Agg' (modo no interactivo) para evitar errores gráficos en entornos web o consola
matplotlib.use('Agg')
# Importación de pyplot para la creación de gráficos (aunque se usa internamente en el módulo)
import matplotlib.pyplot as plt
# Importación de utilidades de scikit-image para lectura (io), conversión de color (color), características (feature), filtros y medidas
from skimage import io, color, feature, filters, measure
# Importación de numpy para operaciones y arreglos numéricos matriciales
import numpy as np

# Definición de la ruta raíz del proyecto de forma absoluta a partir de la ubicación actual del archivo
ROOT = Path(__file__).resolve().parent.parent.parent
# Definición de la carpeta donde se almacenan las imágenes de datos
DATA_DIR = ROOT / "data"
# Definición de la carpeta de artefactos y salidas del sistema
ARTIFACTS_DIR = ROOT / "artifacts"
# Creación automática de la carpeta artifacts si no existe previamente
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Diccionario oficial que asocia el nombre del archivo de imagen con la descripción textual del fallo técnico
FALLOS_DESCRIPCION = {
    "disco_duro_dañado_por_golpe.jpg": "Fisura estructural en el chasis y daño por impacto mecánico directo en los platos internos.",
    "disco_duro_dañado_por_rayones.jpg": "Rayones profundos en la superficie magnética, causando pérdida permanente de sectores de lectura.",
    "disco_duro_no_gira.jpg": "Bloqueo mecánico del eje o fallo en el motor spindle; el disco no genera rotación.",
    "router_falla_conexion_intermitente.jpg": "Inestabilidad en la pasarela, pérdida de paquetes y caídas constantes del enlace inalámbrico.",
    "pantalla_dañada_por_golpe.jpg": "Fractura de matriz LED interna, cristal líquido derramado y líneas muertas en el display.",
    "pantalla_no_arranca.jpg": "Fallo de alimentación en la placa T-CON o ausencia de señal de video (pantalla negra)."
}

def procesar_imagen_proyecto(nombre_imagen="disco_duro.png"):
    """
    Procesa una imagen del proyecto aplicando:
    1. Conversión a escala de grises.
    2. Histograma de intensidad global.
    3. Patrones Locales Binarios (LBP) para texturas.
    4. Umbralización de Otsu y etiquetado (regionprops) filtrando ruido menor a 50 píxeles.
    5. Extracción de características consolidadas en un vector numpy.
    """
    # Construcción de la ruta completa hacia la imagen solicitada
    img_path = DATA_DIR / nombre_imagen
    # Verificación de existencia del archivo; si no existe, lanza un error controlado
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen en {img_path}")

    # Lectura de la imagen desde el disco usando scikit-image
    image_color = io.imread(img_path)
    # Condicional para comprobar si la imagen tiene 3 canales (RGB a color)
    if len(image_color.shape) == 3:
        # Conversión de la imagen a escala de grises para simplificar el procesamiento
        image = color.rgb2gray(image_color)
    else:
        # Si ya está en escala de grises, se asigna directamente
        image = image_color

    # 1. Cálculo del Histograma de Intensidad Global (distribución de tonos de píxeles)
    hist_int, _ = np.histogram(image * 255, bins=32, range=(0, 255), density=True)

    # 2. Configuración para extraer descriptores de textura mediante Patrones Locales Binarios (LBP)
    radius = 1
    n_points = 8 * radius
    # Generación de la matriz LBP evaluando vecindarios uniformes alrededor de cada píxel
    lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
    # Cálculo del histograma de la textura LBP obtenida
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    # 3. Aplicación del método de Otsu para calcular el umbral óptimo de binarización
    thresh = filters.threshold_otsu(image)
    # Creación de la máscara binaria separando el objeto del fondo según el umbral
    binary_mask = image > thresh
    # Etiquetado de los componentes conexos presentes en la máscara binaria
    labels = measure.label(binary_mask)
    # Extracción de propiedades geométricas de cada región etiquetada
    regions = measure.regionprops(labels)

    # Filtrado de regiones pequeñas (consideradas ruido si su área es menor a 50 píxeles)
    areas = [r.area for r in regions if r.area >= 50]
    
    # Verificación si quedaron regiones válidas después del filtrado
    if areas:
        # Cálculo del área media de los componentes encontrados
        area_media = np.mean(areas)
        
        # Cálculo de la desviación estándar de las áreas
        area_std = np.std(areas)
        # Conteo total de regiones significativas
        cantidad_regiones = len(areas)
    else:
        # Si no hay regiones válidas, se inicializan en cero
        area_media = 0.0
        area_std = 0.0
        cantidad_regiones = 0

    # Consolidación de las métricas morfológicas y estadísticas en un vector unidimensional de numpy
    vector_caracteristicas = np.array([area_media, area_std, float(cantidad_regiones)])

    # Retorno de la imagen procesada, máscara binaria, histogramas y el vector de características
    return image, binary_mask, hist_int, hist_lbp, vector_caracteristicas

if __name__ == "__main__":
    # Bloque de ejecución principal para prueba unitaria del módulo independiente
    img, mask, hi, hl, feat = procesar_imagen_proyecto("disco_duro_bueno.jpg")
    print("Prueba de texturas y características ejecutada con éxito.")
    print(f"Vector de características obtenido: {feat}")
# src/classifiers/texturas.py
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para evitar errores gráficos en entornos web o CLI
import matplotlib.pyplot as plt
from skimage import io, color, feature, filters, measure
import numpy as np

# Configuración de rutas absolutas basadas en la estructura del proyecto
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Diccionario oficial con la especificación de los fallos técnicos para la Semana 10
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
    img_path = DATA_DIR / nombre_imagen
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen en {img_path}")

    # Cargar imagen y convertir a escala de grises
    image_color = io.imread(img_path)
    if len(image_color.shape) == 3:
        image = color.rgb2gray(image_color)
    else:
        image = image_color

    # 1. Histograma de Intensidad Global
    hist_int, _ = np.histogram(image * 255, bins=32, range=(0, 255), density=True)

    # 2. Descriptores de Textura LBP (Local Binary Patterns)
    radius = 1
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    # 3. Umbralización (Otsu) y Etiquetado de Regiones
    thresh = filters.threshold_otsu(image)
    binary_mask = image > thresh
    labels = measure.label(binary_mask)
    regions = measure.regionprops(labels)

    # Filtrar regiones pequeñas (ruido menor a 50 píxeles) y extraer áreas
    areas = [r.area for r in regions if r.area >= 50]
    
    if areas:
        area_media = np.mean(areas)
        area_std = np.std(areas)
        cantidad_regiones = len(areas)
    else:
        area_media = 0.0
        area_std = 0.0
        cantidad_regiones = 0

    # Vector consolidado de características morfológicas y estadísticas
    vector_caracteristicas = np.array([area_media, area_std, float(cantidad_regiones)])

    return image, binary_mask, hist_int, hist_lbp, vector_caracteristicas

if __name__ == "__main__":
    # Prueba unitaria del módulo
    img, mask, hi, hl, feat = procesar_imagen_proyecto("disco_duro_bueno.jpg")
    print("Prueba de texturas y características ejecutada con éxito.")
    print(f"Vector de características obtenido: {feat}")
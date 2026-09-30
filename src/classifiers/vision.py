# src/classifiers/vision.py
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Evita errores de interfaz gráfica
import matplotlib.pyplot as plt
from skimage import io, color, feature, filters, measure
import numpy as np

# Configuración de rutas (subiendo 3 niveles desde classifiers)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# La función ahora acepta el parámetro 'nombre_imagen'
def ejecutar_vision_soporte(nombre_imagen="disco_duro.png"):
    print(f"=== INICIANDO PIPELINE DE VISIÓN: SEMANA 09 ({nombre_imagen}) ===")
    
    img_path = DATA_DIR / nombre_imagen
    if not img_path.exists():
        print(f"Error: No se encontró la imagen en {img_path}")
        return None, None

    image_color = io.imread(img_path)
    if len(image_color.shape) == 3:
        image = color.rgb2gray(image_color)
    else:
        image = image_color

    sigma_val = 2.0
    edges = feature.canny(image, sigma=sigma_val)

    threshold = filters.threshold_otsu(image)
    mask = image > threshold 

    labels = measure.label(mask)
    num_regiones = labels.max()

    print(f"Umbral automático Otsu obtenido: {round(threshold, 4)}")
    print(f"Número de regiones conectadas encontradas: {num_regiones}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original (Escala de Grises)")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"Contornos (Canny, sigma={sigma_val})")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Máscara Binaria (Otsu)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path = ARTIFACTS_DIR / "semana09_vision.png"
    fig.savefig(output_path, dpi=160)
    print(f"Evidencia visual guardada exitosamente en: {output_path}")
    print("=== PIPELINE COMPLETADO ===")
    
    return threshold, num_regiones

if __name__ == "__main__":
    ejecutar_vision_soporte()
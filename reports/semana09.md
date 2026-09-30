# Informe Técnico Semana 09: Reconocimiento de Imágenes
**Proyecto:** Asistente de Soporte TI Híbrido
**Estudiante:** Daniel Eduardo Daza Cuello
**Institución:** Escuela Tecnológica Instituto Técnico Central (ETITC)

## Enlace al Repositorio
[https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido](https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido)

https://github.com/MarcoMolina2011/asistente_soporte_TI_hibrido

---
## 1. Imagen utilizada y relación con el proyecto
Se integraron tres imágenes representativas (`disco_duro.png`, `pantalla_azul.png`, `router.png`) que simulan la evidencia visual adjuntada por los usuarios finales al reportar fallas en el portal del Asistente de Soporte TI. Estas imágenes son útiles porque demuestran cómo el sistema preprocesa la información gráfica, extrayendo características morfológicas y separando las piezas de hardware (o los bloques de texto de error) del fondo, transformando la evidencia cruda en matrices numéricas analizables antes de cualquier clasificación.

## 2. Resultado de Canny
La implementación del algoritmo de Canny extrajo con éxito los bordes estructurales de los componentes. En el caso del disco duro dañado, trazó los contornos del plato magnético, el brazo lector y la carcasa; en el router, delineó las antenas y puertos. Esto permite al sistema encontrar los límites físicos entre el objeto de interés y el fondo, ignorando la información irrelevante de color y quedándose únicamente con la topología de la falla.

## 3. Umbral Otsu obtenido
El algoritmo de Otsu calculó umbrales dinámicos (por ejemplo, 0.3574 para el disco duro) basados en el histograma de cada imagen. Este valor matemático permitió generar una máscara binaria que separa automáticamente los píxeles claros (como las partes metálicas reflectantes o el texto blanco) de los oscuros (sombras o fondos lisos), aislando la región de interés sin intervención humana.

## 4. Número de regiones encontradas
A través de la máscara binaria, el sistema identificó múltiples regiones conectadas (ej. 932 regiones en el procesamiento del disco duro). Estas regiones no equivalen a 932 objetos distintos, sino a agrupaciones de píxeles que superaron el umbral de Otsu. Representan fragmentos de circuitos, reflejos de luz sobre el metal, tornillos y divisiones de la carcasa.

## 5. Cambios realizados al modificar sigma
Durante la detección de contornos con Canny, se ajustó el parámetro `sigma` a `2.0`. 
* Un `sigma` menor (ej. 0.5 o 1.0) provocaba que el algoritmo detectara demasiado ruido visual, como polvo, rasguños microscópicos y texturas de la mesa.
* Un `sigma` moderado de `2.0` aplicó el suavizado gaussiano necesario para ignorar el ruido y conservar únicamente las líneas estructurales críticas del hardware.

## 6. Limitaciones encontradas
La segmentación global por Otsu presenta dificultades con la iluminación irregular. Los reflejos fuertes sobre piezas metálicas (como el interior del disco duro) y las sombras proyectadas hacen que el algoritmo fragmente un mismo componente en varias regiones desconectadas. Si el fondo tiene un nivel de intensidad muy similar al del objeto, la máscara binaria fusiona partes del entorno con el hardware.

## 7. Aplicación futura dentro del proyecto
Este pipeline de reconocimiento es la base de la visión computacional del Asistente Híbrido. En futuras iteraciones, las máscaras binarias y los mapas de bordes generados servirán como entrada (input) limpia para una Red Neuronal Convolucional (CNN). Al entregarle a la red imágenes preprocesadas donde ya se eliminó el fondo y el ruido, aumentará drásticamente la precisión del modelo al clasificar automáticamente si la foto corresponde a un daño de red, hardware físico o error de software.
---

## Conclusiones

* **Transformación Numérica del Dato Visual:** Se demostró de manera práctica que un computador no interpreta una imagen de forma semántica directa, sino a través de una matriz de intensidades numéricas que deben ser procesadas por etapas sucesivas.
* **Eficiencia de la Detección de Bordes (Canny):** La aplicación del operador Canny con un parámetro de suavizado ($\sigma = 2.0$) permitió aislar con alta precisión los límites físicos y topológicos de los componentes de hardware (como discos duros y routers), filtrando el ruido visual irrelevante de la escena.
* **Automatización del Umbral (Otsu):** La segmentación mediante el método de Otsu eliminó la subjetividad humana al calcular dinámicamente un umbral óptimo basado en el histograma de la imagen, generando máscaras binarias objetivas que separan el objeto del fondo.
* **Utilidad del Etiquetado de Regiones:** El recuento y etiquetado de regiones conectadas proporcionó una aproximación cuantitativa de los fragmentos, reflejos y componentes internos presentes en la evidencia visual, sirviendo como métrica cuantitativa inicial antes de alimentar modelos de aprendizaje profundo.
* **Escalabilidad hacia un Sistema Híbrido:** Las máscaras y mapas de contornos generados sientan las bases técnicas para extraer características geométricas avanzadas, las cuales optimizarán en fases futuras la precisión diagnóstica del Asistente de Soporte TI al integrarse con modelos predictivos y redes neuronales.
# Informe Técnico - Semana 10: Reconocimiento de Imágenes, Histogramas y Texturas LBP

**Proyecto:** Asistente de Soporte TI Híbrido  
**Institución:** Escuela Tecnológica Instituto Técnico Central (ETITC) - 10º Semestre  
**Autores:** Marco Molina & Daniel Daza  

## Enlace al Repositorio

[https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido](https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido)

[https://github.com/MarcoMolina2011/asistente_soporte_TI_hibrido.git](https://github.com/MarcoMolina2011/asistente_soporte_TI_hibrido.git)
---

## 1. Introducción y Selección de Imágenes
En el contexto del diagnóstico automatizado de soporte técnico, la identificación temprana de fallas físicas en dispositivos es crucial. Para esta práctica de la Semana 10, se seleccionaron pares de imágenes representativas del dominio del proyecto divididas en estado óptimo (sano) y estado con avería específica:
* **Router Óptimo (`router_bueno.jpg`):** Dispositivo de red en perfecto estado estructural, con líneas limpias y superficies planas uniformes.
* **Router Averiado (`router_falla_conexion_intermitente.jpg`):** Dispositivo expuesto a condiciones de desgaste o entornos con iluminación heterogénea que provocan inestabilidad en la pasarela de red y pérdidas de paquetes.

El objetivo principal es analizar y contrastar la distribución global de tonos de gris, las propiedades morfológicas de las regiones conectadas y los patrones locales de textura mediante *Local Binary Patterns (LBP)*.

---

## 2. Segmentación mediante Histogramas y Umbral Otsu
La segmentación permite aislar los elementos relevantes de la imagen respecto al fondo. El pipeline implementado ejecuta los siguientes pasos:
* **Histograma de Intensidad:** Resume la frecuencia de distribución de los niveles de gris (de 0 a 255) en la imagen analizada.
* **Umbral Automático de Otsu:** Calcula matemáticamente un umbral óptimo global minimizando la varianza intraclase. Los píxeles por encima del umbral se separan del fondo.
* **Máscara Binaria:** Matriz booleana resultante que aísla las formas y componentes estructurales del dispositivo frente al entorno.

---

## 3. Etiquetado y Medición de Regiones (`regionprops`)
Mediante las funciones `measure.label()` y `measure.regionprops()` de `scikit-image`, se identifican agrupaciones de píxeles conectados en la máscara binaria:
* **Filtrado de Ruido:** Se descartaron todas las regiones con un área menor a 50 píxeles (`r.area >= 50`) para eliminar imperfecciones menores y ruido visual.
* **Métricas Extraídas:**
  * **Cantidad de Regiones Validadas:** Cuantifica los bloques estructurales detectados (por ejemplo, el router sano muestra regiones consolidadas principales, mientras que el averiado dispersa la segmentación en múltiples áreas debido a sombras e irregularidades físicas).
  * **Área Media:** Tamaño promedio en píxeles de las regiones validadas.
  * **Desviación Estándar del Área:** Mide la dispersión o heterogeneidad en el tamaño de los componentes detectados.

---

## 4. Análisis de Texturas LBP (Local Binary Patterns)
El operador LBP describe la textura local examinando vecindarios circulares de radio $R=1$ con $8$ puntos de muestreo:
* **Comparativa de Texturas:**
  * En el componente óptimo, las superficies planas y uniformes generan respuestas LBP con picos concentrados en patrones homogéneos.
  * En el componente averiado, los cambios abruptos de textura y variaciones de intensidad generan histogramas LBP más dispersos y con múltiples variaciones de frecuencia.

---

## 5. Vector de Características Consolidado (.npy)
Los descriptores obtenidos se unifican en un vector numérico multidimensional utilizando NumPy, que integra:
1. Estadísticas morfológicas de regiones (Área media, desviación estándar y conteo de regiones validadas).
2. Frecuencias del histograma de intensidad global.
3. Frecuencias del histograma de texturas LBP.

Este vector estructurado se almacena localmente en la carpeta de artefactos para alimentar los clasificadores del asistente de soporte.

---

## 6. Limitaciones y Aplicación al Proyecto
* **Limitaciones:** El sistema es sensible a las condiciones de iluminación de la captura fotográfica. Variaciones severas en la luz o sombras proyectadas pueden desplazar el umbral de Otsu y alterar las mediciones morfológicas y los descriptores LBP.
* **Aplicación al Proyecto:** Este módulo dota al Asistente de Soporte TI de la capacidad de realizar un triaje visual preliminar de hardware averiado antes de derivar el caso a soporte especializado o presencial.
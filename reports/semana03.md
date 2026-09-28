# Semana 03: Taxonomía de IA en Entorno Empresarial (Soporte TI)

## Integrantes del Proyecto
* **Marco Molina Molina**
* **Daniel Eduardo Daza Cuello**
* **Institución / Curso:** ETITC - 10º Semestre

## Enlace al Repositorio
[https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido](https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido)

---

## Resultado automático frente a clasificación manual de referencia
| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |
|---|---|---|---|---|
| 1 | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Coincide |
| 2 | Sistemas de Recomendación y Diagnóstico | Sistemas de Recomendación y Diagnóstico, Aprendizaje Automático Predictivo | Sistemas de Recomendación y Diagnóstico | Coincide |
| 3 | Aprendizaje Automático Predictivo | Aprendizaje Automático Predictivo, Búsqueda y Optimización de PC | Aprendizaje Automático Predictivo | Coincide |
| 4 | Búsqueda y Optimización de PC | Búsqueda y Optimización de PC, Sistemas de Recomendación y Diagnóstico | Búsqueda y Optimización de PC | Coincide |
| 5 | Sistemas de Recomendación y Diagnóstico | Sistemas de Recomendación y Diagnóstico, Seguridad y Análisis de Errores (Logs) | Sistemas de Recomendación y Diagnóstico | Coincide |
| 6 | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Coincide |
| 7 | Búsqueda y Optimización de PC | Búsqueda y Optimización de PC, Sistemas Expertos y Reglas de Soporte, Sistemas de Recomendación y Diagnóstico | Búsqueda y Optimización de PC | Coincide |
| 8 | Procesamiento de Lenguaje Natural (PLN) | Procesamiento de Lenguaje Natural (PLN) | Procesamiento de Lenguaje Natural (PLN) | Coincide |
| 9 | Aprendizaje Automático Predictivo | Aprendizaje Automático Predictivo | Aprendizaje Automático Predictivo | Coincide |
| 10 | Sistemas Expertos y Reglas de Soporte | Sistemas Expertos y Reglas de Soporte, Seguridad y Análisis de Errores (Logs), Sistemas de Recomendación y Diagnóstico | Sistemas Expertos y Reglas de Soporte | Coincide |
| 11 | Automatización y Mantenimiento de Software | Automatización y Mantenimiento de Software, Sistemas de Recomendación y Diagnóstico | Automatización y Mantenimiento de Software | Coincide |
| 12 | Procesamiento de Lenguaje Natural (PLN) | Procesamiento de Lenguaje Natural (PLN), Búsqueda y Optimización de PC | Procesamiento de Lenguaje Natural (PLN) | Coincide |
| 13 | Automatización y Mantenimiento de Software | Automatización y Mantenimiento de Software | Automatización y Mantenimiento de Software | Coincide |
| 14 | Sistemas Expertos y Reglas de Soporte | Sistemas Expertos y Reglas de Soporte | Sistemas Expertos y Reglas de Soporte | Coincide |
| 15 | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Coincide |
| 16 | Sistemas de Recomendación y Diagnóstico | Sistemas de Recomendación y Diagnóstico | Sistemas de Recomendación y Diagnóstico | Coincide |
| 17 | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Seguridad y Análisis de Errores (Logs) | Coincide |
| 18 | Sistemas Expertos y Reglas de Soporte | Sistemas Expertos y Reglas de Soporte, Sistemas de Recomendación y Diagnóstico | Sistemas Expertos y Reglas de Soporte | Coincide |
| 19 | Automatización y Mantenimiento de Software | Automatización y Mantenimiento de Software, Búsqueda y Optimización de PC | Automatización y Mantenimiento de Software | Coincide |
| 20 | Búsqueda y Optimización de PC | Búsqueda y Optimización de PC, Sistemas Expertos y Reglas de Soporte | Búsqueda y Optimización de PC | Coincide |

Coincidencia con la referencia: **100.00%** (20/20).

---

## Documentación de la Práctica - Semana 03: Taxonomía de IA en Soporte de Software Empresarial

### 1. Resumen del Módulo
Durante la Semana 03, se implementó el módulo de clasificación y taxonomía inteligente para el Asistente de Soporte TI Híbrido. El sistema procesa de manera automatizada incidencias corporativas categorizándolas en 7 áreas clave mediante reglas léxicas con delimitadores de palabra completa (`\\b`) para prevenir falsos positivos.

### 2. Arquitectura y Componentes Desarrollados
- **Conjunto de Datos (`data/casos_ia.csv`):** 20 casos de prueba de soporte técnico empresarial contextualizados por área.
- **Clasificador Taxonómico (`src/classifiers/taxonomia.py`):** Normalización léxica (minúsculas, remoción de acentos, preservación de espacios) y verificación estricta por palabras completas.
- **Validación de Referencia:** Validación automática contra la pauta de referencia manual (`MANUAL_REFERENCE`), alcanzando el **100.00% de coincidencia** (20 de 20 casos).

### 3. Conclusiones y Métricas
- **Precisión:** Coincidencia perfecta (20/20) al controlar la delimitación de palabras y refinar las reglas departamentales.
- **Trazabilidad:** Integración directa con el flujo central `main.py` y registro persistente en `artifacts/audit.log`.
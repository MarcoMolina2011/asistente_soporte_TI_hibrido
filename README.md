# Asistente de Soporte TI Híbrido

Sistema inteligente de soporte técnico diseñado para automatizar la clasificación de consultas, recuperar conocimiento técnico, aplicar reglas de negocio y priorizar incidentes garantizando trazabilidad completa.
---

##  Arquitectura del Sistema

El proyecto combina un enfoque modular y híbrido adaptado a las necesidades de soporte de software de PC corporativo:
- **Semana 09 - Visión Artificial y Procesamiento de Imágenes:**
Se implementó el pipeline de preprocesamiento y reconocimiento visual para transformar evidencia gráfica en datos numéricos estructurados:
* **Extracción de Características & Contornos (Canny):** Aplicación de filtros gaussianos ($\sigma = 2.0$) y detección de gradientes para aislar los límites estructurales de los componentes de hardware reportados[cite: 2].
* **Segmentación por Umbrales (Otsu):** Cálculo automatizado del umbral óptimo basado en el histograma para separar los objetos de interés del fondo mediante una máscara binaria[cite: 2].
* **Análisis de Regiones Conectadas:** Etiquetado y conteo cuantitativo de grupos de píxeles para caracterizar los elementos presentes en la evidencia visual[cite: 2].
* **Evidencia e Informe:** Artefactos generados en `artifacts/semana09_vision.png` y documentación técnica completa en `reports/semana09.md`.
- **Semana 08 - Red Neuronal, Evidencia y Ontología:** 
  * *Red Neuronal (MLP):* Clasificador de telemetría e incidentes basado en un Perceptrón Multicapa.
  * *Base de Evidencia (SQLite):* Almacenamiento persistente y auditable de cada inferencia y metadato del modelo en `artifacts/soporte_evidencia.db`.
  * *Ontología (GraphML):* Representación formal del conocimiento mediante grafos en NetworkX (`artifacts/ontologia.graphml`), conectando la predicción con su respectivo significado semántico dentro del dominio.
- **Semana 07 - Representaciones del Reconocimiento:** Integración de tres enfoques para la interpretación de fallas:
1. **Numérica:** Análisis de telemetría mediante distancias euclidianas para detectar riesgos de colapso en servidores.
2. **Simbólica:** Sistema experto con inferencia lógica para diagnósticos a partir de hechos discretos.
3. **Autómatas:** Autómata Finito Determinista (AFD) para auditar secuencias de logs y detectar fallas en cascada.
- **Sistema Híbrido y Base de Conocimiento (Semana 05):** Integración de un motor híbrido que combina reglas expertas deterministas, vectorización TF-IDF y similitud coseno sobre una base de conocimiento técnica de 30 entradas, con generación automatizada de informes en Markdown (`reports/semana05.md`).
- **Marco Tecnológico y Búsqueda Inteligente (Semana 04):** Incorpora algoritmos de búsqueda y optimización avanzada, incluyendo el planificador **$A^*$** para secuencias óptimas de diagnóstico de soporte y **Minimax** para la toma de decisiones en entornos de recursos limitados.
- **Recuperador (RAG / Base de Conocimiento):** Busca soluciones y documentación técnica previa asociada a los errores reportados.
- **Clasificador Taxonómico (Semana 03):** Categoriza automáticamente las incidencias corporativas en 7 áreas clave (PLN, Aprendizaje Predictivo, Optimización de PC, Sistemas Expertos, Automatización, Seguridad/Logs y Diagnóstico) considerando el impacto por departamento (Contabilidad, Recursos Humanos, Ventas, Logística, Gerencia, etc.).
- **Recuperador (RAG / Base de Conocimiento):** Busca soluciones y documentación técnica previa asociada a los errores reportados.
- **Motor de Reglas y Priorización (Semana 02):** Aplica políticas empresariales combinando Urgencia e Impacto para definir prioridades de atención.
- **Trazabilidad y Auditoría (Semana 02):** Registra cada acción, evento y cambio de estado en un sistema persistente estructurado (`artifacts/audit.log`).
---

## Stack Tecnológico

| Herramienta | Uso en el Proyecto |
| :--- | :--- |
| **Python 3.13+** | Lenguaje base del motor lógico y orquestador (`main.py`). |
| **Streamlit** | Framework para la construcción del Dashboard interactivo. |
| **Scikit-learn** | Clasificación de texto, vectorización (TF-IDF) y similitud coseno. |
| **Pandas / NumPy** | Procesamiento matemático, operaciones vectoriales y DataFrames. |
| **Pydantic** | Validación estricta de esquemas y estructura de tickets. |

---

##  Instalación y Configuración
Sigue estos pasos para desplegar el proyecto en tu entorno local:

1. **Clonar el repositorio:**
    ```bashbash
   git clone [https://github.com/DanielDaza2901/asistente_soporte_ti.git](https://github.com/DanielDaza2901/asistente_soporte_ti.git)
   cd asistente_soporte_ti

2. **Instalar dependencias:**
(Se recomienda el uso de un entorno virtual .venv)
     ```bash
   pip install scikit-learn==1.5.0
   pip install -r requirements.txt
     ```

3. **Ejecutar la orquestación en consola (Flujo Completo):**
     ```bash
     python src/main.py
    ```
4. **Ejecutar el Dashboard Interactivo:**
     ```bash
    streamlit run dashboard.py
    ```

---

##  Estructura del Proyecto

```text
asistente_soporte_ti/
├── artifacts/          # Logs de auditoría y trazas del sistema (audit.log)
├── data/               # Conjuntos de datos y Base de Conocimiento (KB)
├── notebooks/          # Notebooks de experimentación y análisis
├── reports/            # Informes automáticos generados (.md)
├── src/                # Código fuente principal
│   ├── audit/          # Módulo de trazabilidad y logging
│   ├── classifiers/    # Taxonomía, A*, Minimax, y Sistema Híbrido
│   ├── knowledge/      # Gestión de conocimiento técnico
│   ├── rules/          # Motor de reglas y priorización ITIL
│   ├── semana07_representaciones.py # Módulo de representaciones (Numérica/Simbólica/Autómatas)
│   └── main.py         # Orquestador principal del flujo del asistente en consola
├── tests/              # Pruebas unitarias e integración
├── dashboard.py        # Interfaz web interactiva (Streamlit)
├── requirements.txt    # Dependencias del entorno
└── README.md           # Documentación general del proyecto
```
---


##  Autores
- Estudiantes: - Marco Molina Molina
               - Daniel Eduardo Daza Cuello
- Institución: ETITC - 10º Semestre

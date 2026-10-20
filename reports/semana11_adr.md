# ADR (Architecture Decision Record) - Semana 11: Benchmark y Selección de Modelo

**Proyecto:** Asistente de Soporte TI Híbrido  
**Institución:** Escuela Tecnológica Instituto Técnico Central (ETITC) - 10º Semestre  
**Autores:** Marco Molina & Daniel Daza  

---

## 1. Contexto y Problema
El desarrollo del Asistente de Soporte TI Híbrido requiere integrar un componente de clasificación automatizada para el triaje y priorización de incidentes técnicos. Para evitar elecciones arbitrarias, este documento registra la justificación técnica basada en la medición empírica, evaluando tres alternativas algorítmicas bajo un protocolo estricto y reproducible.

## 2. Metodología y Protocolo de Evaluación
Para cumplir rigurosamente con los criterios de evaluación, se establecieron los siguientes controles experimentales:
* **Mismo Dataset:** Se utilizó el conjunto de datos de referencia procesado bajo el mismo preprocesamiento estándar (`load_digits`).
* **Misma División de Datos:** Se aplicó `train_test_split` con una proporción de prueba del 25%, semilla fija (`random_state=42`) y estratificación (`stratify=y`) para asegurar que cada modelo fuera evaluado exactamente con las mismas muestras.
* **Métricas Registradas:** Se midió la precisión de generalización (`Accuracy`) y el costo computacional de entrenamiento (`fit_seconds`).

## 3. Modelos Evaluados y Evidencia Empírica
Los tres modelos funcionales implementados obtuvieron los siguientes resultados registrados en `artifacts/benchmark.csv`:

| Modelo | Accuracy | Tiempo de Entrenamiento (s) |
| :--- | :--- | :--- |
| **MLPClassifier (Red Neuronal)** | 0.9800 | 1.452 |
| **Logistic Regression (Línea Base)** | 0.9778 | 0.062 |
| **Random Forest (Ensamble)** | 0.9667 | 0.930 |

## 4. Análisis Analítico de la Gráfica Comparativa
A partir de la visualización de los resultados en el dashboard, se identifican las siguientes dinámicas de rendimiento:
1. **Rendimiento de la Línea Base:** *Logistic Regression* confirma que el problema de clasificación cuenta con fuerte tendencia lineal, logrando un notable nivel de accuracy con un costo computacional mínimo.
2. **Superioridad del Modelo Ganador:** *MLPClassifier (Red Neuronal)* maximiza el acierto predictivo alcanzando **98.00%**, lo cual reduce al mínimo los falsos positivos en el triaje de tickets críticos de Nivel 1.
3. **Análisis de Costo-Beneficio:** Aunque el tiempo de entrenamiento del modelo seleccionado es de **1.452 segundos**, este costo es despreciable dado que el entrenamiento se realiza de forma offline, priorizando una alta precisión en la inferencia en tiempo real.

## 5. Decisión Técnica y Justificación
* **Modelo Seleccionado:** **MLPClassifier (Red Neuronal)**.
* **Justificación de la Elección:** Se priorizó el balance entre la máxima tasa de generalización y la estabilidad ante variaciones de ruido en los reportes de incidentes. El artefacto ha sido serializado en `artifacts/mejor_modelo.pkl`.

## 6. Ventajas, Limitaciones y Aplicación al Proyecto Individual
* **Ventajas:** Alta confiabilidad en la categorización automatizada de tickets de soporte.
* **Limitaciones:** Requiere reentrenamiento periódico ante la aparición de nuevas tipologías de fallas tecnológicas.
* **Aplicación al Proyecto Final:** Integración directa en el backend y el flujo de la interfaz del asistente de soporte.

## 7. Conclusión de Ingeniería
El pipeline se ejecutó satisfactoriamente sin errores, generando las evidencias persistentes en `artifacts/benchmark.csv`, el modelo serializado, el registro de auditoría en `audit.log` y el presente informe ADR, cumpliendo a cabalidad con la trazabilidad requerida en el repositorio GitHub.

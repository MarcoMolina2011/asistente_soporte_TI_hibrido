# Informe Técnico - Semana 13: Ingeniería del Conocimiento y Motor de Inferencia Lógica

**Proyecto:** Asistente de Soporte TI Híbrido  
**Institución:** Escuela Tecnológica Instituto Técnico Central (ETITC) - 10º Semestre  
**Autores:** Marco Molina & Daniel Daza  
**Fecha de Entrega:** Noviembre de 2026  

---
## Enlace al Repositorio
[https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido](https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido)

https://github.com/MarcoMolina2011/asistente_soporte_TI_hibrido
---

## 1. Análisis del Dominio
El módulo desarrollado en la Semana 13 aborda la automatización del **triaje y diagnóstico de incidentes críticos de infraestructura TI**, conectando la evidencia visual obtenida en semanas anteriores (procesamiento de imágenes, texturas LBP y umbrales Otsu) con un sistema lógico formal.

* **Problema que se desea resolver:** Determinar de forma automatizada y auditable el nivel de escalamiento y prioridad de un ticket técnico a partir de hechos discretos observados en los dispositivos.
* **Conceptos Principales:** Integridad de hardware, procesamiento de imágenes, acuerdos de nivel de servicio (SLA), enlaces WAN y escalamiento jerárquico de soporte.
* **Hechos Iniciales (Vocabulario Controlado):**
  * `imagen_procesada_ok`: La captura de diagnóstico fue procesada correctamente.
  * `textura_anomala_detectada`: Se identificaron anomalías físicas o patrones de daño en la superficie del componente.
  * `hardware_critico`: El elemento pertenece al núcleo de procesamiento o almacenamiento principal.
  * `sla_excedido`: El tiempo límite de resolución estipulado ha sido superado.
  * `conexion_intermitente`: Inestabilidad detectada en la pasarela de red.
  * `alta_latencia`: Demoras críticas en la red WAN.
  * `impacto_critico`: Afectación sistémica a múltiples usuarios de la corporación.
* **Conclusión Esperada:** Diagnóstico definitivo de fallo físico, derivación a nivel de soporte superior o clasificación como `prioridad_critica_urgente`.

---

## 2. Reglas Lógicas (Lógica Proposicional y Reglas de Horn)
Se implementaron 5 reglas originales estructuradas bajo la forma de implicación estricta ($\text{Premisas} \longrightarrow \text{Conclusión}$):

1. **Regla R1:** 
   $$\text{imagen\_procesada\_ok} \land \text{textura\_anomala\_detectada} \longrightarrow \text{diagnostico\_hardware\_fallido}$$
   *Descripción:* Si la imagen se procesó con éxito y muestra anomalías de textura, se confirma un fallo físico de hardware.

2. **Regla R2:** 
   $$\text{diagnostico\_hardware\_fallido} \land \text{hardware\_critico} \longrightarrow \text{escalamiento\_nivel\_2}$$
   *Descripción:* Si hay un fallo de hardware confirmado en un componente crítico, el ticket se escala al Nivel 2.

3. **Regla R3:** 
   $$\text{escalamiento\_nivel\_2} \land \text{sla\_excedido} \longrightarrow \text{prioridad\_critica\_urgente}$$
   *Descripción:* Si el ticket escalado a Nivel 2 excede su SLA, se clasifica como prioridad crítica y urgente.

4. **Regla R4:** 
   $$\text{conexion\_intermitente} \land \text{alta\_latencia} \longrightarrow \text{falla\_enlace\_wan}$$
   *Descripción:* Si se reporta conectividad intermitente y alta latencia, se deduce una falla general de enlace WAN.

5. **Regla R5:** 
   $$\text{falla\_enlace\_wan} \land \text{impacto\_critico} \longrightarrow \text{escalamiento\_redes\_nivel\_3}$$
   *Descripción:* Si la falla de enlace WAN genera impacto crítico corporativo, se deriva a ingeniería de redes Nivel 3.

---

## 3. Arquitectura del Motor de Inferencia (`src/motor_inferencia.py`)
El motor emplea **encadenamiento hacia adelante (*forward chaining*)** incorporando los siguientes mecanismos de control:
* **Estructura Inmutable:** Definición de la clase `Regla` mediante contenedores congelados (`FrozenSet`).
* **Prevención de Ciclos y Duplicados:** Validación rigurosa (`conclusion not in hechos`) para evitar bucles infinitos en caso de dependencias circulares y asegurar que las conclusiones se añadan una sola vez.
* **Explicabilidad Paso a Paso:** Cada deducción genera un registro detallado de Modus Ponens que indica el ciclo, la regla disparada y las premisas utilizadas.

---

## 4. Pruebas Funcionales y Validación (`tests/test_motor_inferencia.py`)
Se validaron formalmente los requerimientos de la actividad mediante pruebas unitarias con `unittest`:
* **Caso A (Conclusión Demostrada):** Comprobación exitosa al ingresar evidencias completas (`["R1", "R2", "R3"]`), alcanzando la meta `prioridad_critica_urgente`.
* **Caso B (Conclusión No Demostrada):** Demostración al omitir el SLA excedido. El sistema deriva hasta `escalamiento_nivel_2` pero se detiene. *(Nota pedagógica: "No demostrada" no significa falsa, sino que los hechos de entrada fueron insuficientes).*
* **Conservación de Hechos:** Verificación de que el conjunto de hechos iniciales no sea mutado durante la ejecución.
* **Ausencia de Ciclos Infinitos:** Prueba con dependencias circulares artificiales superada exitosamente gracias al control de conclusiones nuevas.

---

## 5. Verificación de Cumplimiento (Rúbrica de Asignación)

| Requerimiento de la Asignación | Estado | Evidencia en el Repositorio |
| :--- | :---: | :--- |
| **Análisis del dominio** | Cumplido | Definido en este informe y orquestador |
| **Mínimo 3 reglas originales** | Cumplido | 5 reglas implementadas en `src/motor_inferencia.py` |
| **Motor de inferencia en Python** | Cumplido | Implementado con encadenamiento hacia adelante |
| **Prevención de duplicados / ciclos** | Cumplido | Validaciones nativas en el bucle de inferencia |
| **Pruebas funcionales (Caso A y B)** | Cumplido | Cobertura en `tests/test_motor_inferencia.py` |
| **Uso de `unittest`** | Cumplido | Casos automatizados ejecutables |
| **Documentación y Reporte** | Cumplido | Archivo `reports/reporte_semana13.md` |
| **Sustentación y Trazabilidad** | Cumplido | Integrado en terminal, consola y `dashboard.py` |
| **Commit obligatorio en GitHub** | Cumplido | Sincronizado en el repositorio |

---

## 6. Conclusiones y Limitaciones
El motor de inferencia proposicional integrado aporta explicabilidad y rigor analítico al Asistente de Soporte TI. Como limitación principal se destaca que este modelo lógico maneja certeza booleana estricta, por lo que no contempla de forma nativa la incertidumbre probabilística (la cual se complementa en el proyecto mediante los modelos de Machine Learning de la Semana 11).
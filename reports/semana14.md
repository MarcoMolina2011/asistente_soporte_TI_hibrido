## Semana 14 – Ingeniería del conocimiento: redes semánticas

Red semántica aplicada al **Asistente de Soporte TI Híbrido**. Interpreta oraciones simples en español, las convierte en triples `(sujeto, relación, objeto)`, las almacena como un grafo dirigido y obtiene conclusiones por jerarquía (`es_un`) y herencia de propiedades, mostrando **cómo** se llegó a cada conclusión.

---
   ### Enlace al repositorio
   - Daniel Daza: [asistente_soporte_TI_hibrido](https://github.com/DanielDaza2901/asistente_soporte_TI_hibrido)
   - Marco Molina: [asistente_soporte_TI_hibrido](https://github.com/MarcoMolina2011/asistente_soporte_TI_hibrido)
---
### 1. Conceptos del proyecto (8)

| Concepto | Significado en el proyecto |
|---|---|
| `servidor_bd` | Servidor de base de datos de la infraestructura atendida por soporte |
| `estacion_trabajo` | Equipo de escritorio de un usuario |
| `hardware_critico` | Categoría de equipos cuya falla detiene la operación |
| `hardware_estandar` | Categoría de equipos de uso común |
| `activo_corporativo` | Categoría general de bienes de la organización |
| `alimentacion_redundante` | Requisito de energía de respaldo |
| `sobrecalentamiento` | Falla recurrente del Sistema Híbrido (Semana 05: "equipo caliente") |
| `limpieza_ventilacion` | Procedimiento técnico que corrige el sobrecalentamiento |

### 2. Relaciones

| Relación | Significado | Transitiva | Se hereda |
|---|---|---|---|
| `es_un` | El sujeto es una subcategoría o instancia del objeto | Sí | – |
| `requiere` (sinónimo: `necesita`) | El sujeto necesita el objeto para operar | No | Sí, por `es_un` |
| `se_resuelve_con` | El sujeto se corrige aplicando el objeto | No | No |

Triples del dominio: `(servidor_bd, es_un, hardware_critico)`, `(hardware_critico, es_un, activo_corporativo)`, `(activo_corporativo, requiere, alimentacion_redundante)`, `(estacion_trabajo, es_un, hardware_estandar)`, `(sobrecalentamiento, se_resuelve_con, limpieza_ventilacion)`.

### 3. Reglas de inferencia

* **R1 – Transitividad:** si `A es_un B` y `B es_un C` ⇒ `A es_un C`.
* **R2 – Herencia:** si `A es_un B` y `B requiere P` ⇒ `A requiere P`.

Las conclusiones **no se almacenan**: se calculan recorriendo el grafo con una búsqueda en anchura. Un conjunto de visitados evita ciclos y devuelve la ruta más corta como prueba. `se_resuelve_con` no se hereda porque un procedimiento válido para una categoría no tiene por qué servir para toda subcategoría.

### 4. Interpretación de oraciones

Se normaliza el texto (minúsculas, sin tildes, sin artículo inicial, `_` entre palabras) y se reconocen cuatro formatos: `<A> es un/una <B>`, `<A> requiere <B>`, `<A> necesita <B>` y `<A> se resuelve con <B>`. Se lanza `ValueError` si no coincide ningún formato, si el concepto es vacío, tiene más de 3 palabras o mezcla otra relación (`"x es un y que requiere z"`).


### 5. Resultados (salida de la sección 12)

```text
[12.3] Inferencia por jerarquía (R1: A es_un B y B es_un C => A es_un C):
  ¿servidor_bd es un activo_corporativo? True
  Justificación: Inferido por transitividad (R1): (servidor_bd es_un hardware_critico) + (hardware_critico es_un activo_corporativo) => (servidor_bd es_un activo_corporativo)
  Hecho directo almacenado: False  (la conclusión no se guardó, se infirió)

[12.4] Herencia de propiedades (R2: A es_un B y B requiere P => A requiere P):
  Propiedades de servidor_bd: ['alimentacion_redundante']
  alimentacion_redundante: heredada (R2) de 'activo_corporativo' por la cadena servidor_bd -> hardware_critico -> activo_corporativo.

[12.5] Consultas sin evidencia suficiente (mundo cerrado: False = no demostrado):
  ¿estacion_trabajo es un hardware_critico? False
  Sin evidencia: no existe una cadena es_un de 'estacion_trabajo' a 'hardware_critico' en los hechos registrados.
  Propiedades de estacion_trabajo: []

[12.6] Validación de oraciones no admitidas:
  RECHAZADA: "el procesador funciona muy lento hoy"
    ValueError: Oración no admitida: ... Formatos válidos: '<A> es un <B>', '<A> requiere <B>' (o 'necesita'), '<A> se resuelve con <B>'.
```

| Caso demostrado | Resultado |
|---|---|
| Relación explícita | `(servidor_bd, es_un, hardware_critico)` almacenado |
| Conclusión inferida | `servidor_bd es_un activo_corporativo` (2 pasos, sin arista directa) |
| Herencia | `servidor_bd` requiere `alimentacion_redundante` (vía `activo_corporativo`) |
| Sin evidencia | `estacion_trabajo es_un hardware_critico` → `False` |
| Entrada inválida | `ValueError` y la red no se modifica |

### 6. Limitaciones

* **Mundo cerrado:** `False` significa "no demostrado con los hechos cargados", no "falso".
* El analizador solo reconoce los formatos de la sección 4: no entiende negaciones, ambigüedad ni lenguaje natural general.
* Normalizar pierde matices: `ñ` pasa a `n` (año y ano son el mismo concepto).
* No hay detección de contradicciones ni de ciclos al insertar: los ciclos se toleran en las consultas gracias al conjunto de visitados.

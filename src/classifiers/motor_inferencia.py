# src/motor_inferencia.py
from dataclasses import dataclass
from typing import Set, List, Tuple, FrozenSet, Dict, Any

@dataclass(frozen=True)
class Regla:
    """
    Estructura inmutable que representa una regla lógica de Horn (Premisas -> Conclusión).
    """
    nombre: str
    premisas: FrozenSet[str]
    conclusion: str
    descripcion: str

def inferir_con_explicacion(hechos_iniciales: Set[str], reglas: List[Regla]) -> Tuple[Set[str], List[str], List[Dict[str, Any]]]:
    """
    Motor de inferencia mediante encadenamiento hacia adelante con explicabilidad paso a paso.
    
    Parámetros:
        hechos_iniciales (Set[str]): Proposiciones iniciales verdaderas en el caso.
        reglas (List[Regla]): Catálogo de reglas del dominio.
        
    Retorna:
        Tuple[Set[str], List[str], List[Dict[str, Any]]]: 
            - Conjunto ampliado de hechos deducidos.
            - Lista de trazas (nombres de reglas disparadas).
            - Historial detallado de pasos de derivación para explicabilidad.
    """
    hechos = set(hechos_iniciales)
    trazas = []
    pasos_explicacion = []
    cambios = True
    ciclo_contador = 0
    
    while cambios and ciclo_contador < 100:  # Límite de seguridad contra bucles infinitos
        cambios = False
        ciclo_contador += 1
        
        for regla in reglas:
            # Comprobar si todas las premisas están presentes y la conclusión es nueva
            if regla.premisas.issubset(hechos) and regla.conclusion not in hechos:
                hechos.add(regla.conclusion)
                trazas.append(regla.nombre)
                
                # Registrar paso detallado para explicabilidad
                pasos_explicacion.append({
                    "ciclo": ciclo_contador,
                    "regla": regla.nombre,
                    "descripcion": regla.descripcion,
                    "premisas_usadas": list(regla.premisas),
                    "conclusion_derivada": regla.conclusion
                })
                
                cambios = True
                
    return hechos, trazas, pasos_explicacion

# Catálogo ampliado de reglas originales para el Asistente de Soporte TI (5 Reglas)
REGLAS_SOPORTE_TI = [
    Regla(
        nombre="R1",
        premisas=frozenset({"imagen_procesada_ok", "textura_anomala_detectada"}),
        conclusion="diagnostico_hardware_fallido",
        descripcion="Si la imagen se procesó con éxito y muestra anomalías de textura, se confirma un fallo físico de hardware."
    ),
    Regla(
        nombre="R2",
        premisas=frozenset({"diagnostico_hardware_fallido", "hardware_critico"}),
        conclusion="escalamiento_nivel_2",
        descripcion="Si hay un fallo de hardware confirmado en un componente crítico, el ticket se escala al Nivel 2."
    ),
    Regla(
        nombre="R3",
        premisas=frozenset({"escalamiento_nivel_2", "sla_excedido"}),
        conclusion="prioridad_critica_urgente",
        descripcion="Si el ticket escalado a Nivel 2 excede su SLA, se clasifica como prioridad crítica y urgente."
    ),
    Regla(
        nombre="R4",
        premisas=frozenset({"conexion_intermitente", "alta_latencia"}),
        conclusion="falla_enlace_wan",
        descripcion="Si se reporta conectividad intermitente y alta latencia, se deduce una falla general de enlace WAN."
    ),
    Regla(
        nombre="R5",
        premisas=frozenset({"falla_enlace_wan", "impacto_critico"}),
        conclusion="escalamiento_redes_nivel_3",
        descripcion="Si la falla de enlace WAN genera impacto crítico corporativo, se deriva a ingeniería de redes Nivel 3."
    )
]

if __name__ == "__main__":
    print("=== DEMOSTRACIÓN DEL MOTOR DE INFERENCIA MEJORADO ===")
    hechos_demo = {"imagen_procesada_ok", "textura_anomala_detectada", "hardware_critico", "sla_excedido"}
    finales, trazas, explicacion = inferir_con_explicacion(hechos_demo, REGLAS_SOPORTE_TI)
    print(f"Hechos deducidos: {finales}")
    print(f"Trazas: {trazas}")
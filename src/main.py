# 1. Importación de módulos internos del proyecto (Auditoría, Reglas, Clasificadores e IA)
from audit.logger_audit import registrar_traza
from rules.prioritizer import calcular_prioridad
from classifiers.taxonomia import classify_problem
from classifiers.evaluacion_modelo import ejecutar_validacion
from classifiers.astar import astar_soporte_ti, START_STATE, GOAL_STATE
from classifiers.minimax import best_move, board as minimax_board, simular_ciberdefensa
from classifiers.sistema_hibrido import SistemaHibridoSoporte, generar_reporte
from classifiers.representaciones import ejecutar_representaciones_hibridas
from classifiers.reconocimiento import ejecutar_semana_08
from classifiers.vision import ejecutar_vision_soporte

def main():
    # === SECCIÓN 1: Validación del Modelo Base ===
    print("=== 1. VALIDACIÓN DEL MODELO BASE (MATRIZ DE CONFUSIÓN) ===")
    ejecutar_validacion()
    
    print("\n" + "="*50 + "\n")
    
    # === SECCIÓN 2: Triage y Clasificación de Ticket ===
    ticket_id = "TICK-2026-001"
    descripcion = "El departamento de Contabilidad reporta que la aplicacion de nomina se cierra inesperadamente al generar el informe fiscal mensual."
    impacto = "Alto"
    urgencia = "Alto"

    print(f"=== 2. PROCESANDO TICKET DE SOPORTE: {ticket_id} ===")
    registrar_traza(ticket_id, "RECEPCION", f"Ticket recibido con descripción: '{descripcion}'")

    categoria_principal, categorias_detectadas, _ = classify_problem(descripcion)
    registrar_traza(ticket_id, "TAXONOMIA", f"Categoría principal: '{categoria_principal}' | Detectadas: {categorias_detectadas}")

    prioridad = calcular_prioridad(impacto, urgencia)
    registrar_traza(ticket_id, "PRIORIZACION", f"Impacto: {impacto}, Urgencia: {urgencia} -> Asignada Prioridad: {prioridad}")

    print(f"\nResultado del análisis:")
    print(f"- Categoría de Software (IA): **{categoria_principal}**")
    print(f"- Prioridad asignada: **{prioridad}**")
    print(f"- Traza guardada correctamente en artifacts/audit.log.")

    print("\n" + "="*50 + "\n")
    
    # === SECCIÓN 3: Planificación de Secuencia de Solución con A* ===
    print("=== 3. PLANIFICADOR DE SOPORTE TI CON A* (SEMANA 4) ===")
    print(f"Estado Inicial: {START_STATE} -> Estado Meta: {GOAL_STATE}")
    ruta, costo_total = astar_soporte_ti(START_STATE, GOAL_STATE)
    print(f"Costo acumulado mínimo de resolución: {costo_total}")
    print("Secuencia óptima de acciones técnicas:")
    if ruta:
        for idx, (orig, dest, desc, c) in enumerate(ruta, 1):
            print(f"  {idx}. [{desc}] (Costo: {c}) | Transición: {orig} -> {dest}")
    else:
        print("No se encontró una ruta válida.")

    print("\n" + "="*50 + "\n")
    
    # === SECCIÓN 4: Ciberdefensa Adversarial con Minimax ===
    print("=== 4. CIBERDEFENSA Y DECISIÓN ADVERSARIAL CON MINIMAX (SEMANA 4) ===")
    print(f"Estado de red actual (Matriz 3x3): {minimax_board}")
    diag_minimax = simular_ciberdefensa(minimax_board)
    print(f"Mejor posición estratégica seleccionada por Minimax: {diag_minimax['posicion']} ({diag_minimax['nodo']})")
    print(f"Acción defensiva recomendada: {diag_minimax['accion']}")
    print(f"Justificación técnica: {diag_minimax['justificacion']}")

    print("\n" + "="*50 + "\n")
    
    # === SECCIÓN 5: Motor Híbrido (Reglas + TF-IDF + ML + RAG) ===
    print("=== 5. SISTEMA HÍBRIDO E INFORMES DE CONOCIMIENTO (SEMANA 05) ===")
    sistema = SistemaHibridoSoporte()
    
    pruebas = [
        "El equipo esta muy caliente y el ventilador hace ruido",
        "La conexion de internet cae constantemente y falla el enlace",
        "El disco duro esta lleno y la aplicacion esta muy lenta"
    ]
    
    resultados = [sistema.procesar(p) for p in pruebas]
    
    for idx, r in enumerate(resultados, start=1):
        print(f"\n--- Prueba Híbrida {idx} ---")
        print(f"Consulta:   {r['consulta']}")
        print(f"Reglas:     {r['reglas']}")
        print(f"Evidencia:  {r['evidencia']}")
        print(f"Similitud:  {r['similitud']:.4f}")
        print(f"Clase:      {r['clase']}")
        
    generar_reporte(resultados)
    registrar_traza(ticket_id, "SISTEMA_HIBRIDO", "Ejecución completa del sistema híbrido de la Semana 05 con base de conocimiento.")
    
    print("\n" + "="*50 + "\n")

    # === SECCIÓN 6: Representaciones del Reconocimiento (Semana 07) ===
    print("=== 6. REPRESENTACIONES DEL RECONOCIMIENTO (SEMANA 07) ===")
    ejecutar_representaciones_hibridas()
    registrar_traza(ticket_id, "REPRESENTACIONES", "Ejecución de representaciones numérica, simbólica y de autómata (Semana 07) completada.")
    
    print("\n" + "="*50 + "\n")

    # === SECCIÓN 7: Red Neuronal, Imagen Base64, SQLite y Ontología (Semana 08) ===
    print("=== 7. RED NEURONAL, IMAGEN BASE64, EVIDENCIA SQLite Y ONTOLOGÍA GraphML (SEMANA 08) ===")
    ejecutar_semana_08()
    registrar_traza(ticket_id, "SEMANA_08", "Ejecución del clasificador MLP, persistencia en SQLite de imagen Base64 y ontología GraphML completada.")
    
    print("\n" + "="*50 + "\n")
    
    # === SECCIÓN 8: Visión Artificial (Semana 09) ===
    print("=== 8. VISIÓN ARTIFICIAL (SEMANA 09) ===")
    ejecutar_vision_soporte("disco_duro.png")
    registrar_traza(ticket_id, "SEMANA_09", "Ejecución de pipeline de visión artificial completada.")

    print("==================================================")

if __name__ == "__main__":
    main()
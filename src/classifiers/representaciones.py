# Importa la librería NumPy para realizar operaciones numéricas y matriciales eficientes
import numpy as np

def ejecutar_representaciones_hibridas():
    """
    Ejecuta las tres representaciones del reconocimiento adaptadas al
    Asistente de Soporte TI Híbrido (Triaje numérico, Diagnóstico simbólico y Autómata de Fallas en Cascada).
    """
    
    print("=== 1. REPRESENTACIÓN NUMÉRICA (Triaje Híbrido de Tickets) ===")
    # Vector de características: [Impacto de Infraestructura (1-10), Urgencia de Tiempo (1-10), Sentimiento Negativo NLP (0.0-1.0)]
    ticket_entrada = np.array([8.5, 9.0, 0.95])  
    perfil_critico = np.array([10.0, 10.0, 1.0]) 
    
    # Calcula la distancia euclidiana entre el vector del ticket actual y el perfil crítico predefinido
    distancia = np.linalg.norm(ticket_entrada - perfil_critico)
    
    # Imprime en consola los valores numéricos de los vectores comparados
    print(f"Vector Ticket Actual: {ticket_entrada}")
    print(f"Vector Perfil Crítico: {perfil_critico}")
    print(f"Distancia numérica (similitud a escalamiento inmediato): {round(float(distancia), 3)}")
    
    # Evalúa si la distancia numérica es menor al umbral crítico establecido (< 3.0)
    if distancia < 3.0:
        print("-> Alerta de Triaje: Clasificación prioritaria. Asignando directamente a Especialista Humano.\n")


    print("=== 2. REPRESENTACIÓN SIMBÓLICA (Enrutamiento y Diagnóstico General) ===")
    # Define un conjunto (set) con los hechos discretos detectados por el asistente en el ticket
    hechos_ticket = {"pantalla_azul", "reinicio_constante", "codigo_stop", "equipo_portatil"}
    print(f"Hechos detectados por el Asistente: {hechos_ticket}")
    
    # Define reglas expertas basadas en subconjuntos de hechos simbólicos
    regla_hardware_critico = {"pantalla_azul", "reinicio_constante"}
    regla_accesos = {"olvido_contrasena", "active_directory"}
    
    # Verifica mediante lógica de conjuntos si la regla de hardware crítico está contenida en los hechos del ticket
    if regla_hardware_critico.issubset(hechos_ticket):
        print("-> Conclusión simbólica: diagnostico_falla_hardware_fisico")
        print("-> Acción recomendada: Desviar al módulo de Soporte de Campo (Presencial).\n")
    # Verifica de forma alternativa si aplica la regla de recuperación de credenciales
    elif regla_accesos.issubset(hechos_ticket):
        print("-> Conclusión simbólica: diagnostico_restablecimiento_credenciales")
        print("-> Acción recomendada: Ejecutar script de automatización de reseteo (Nivel 1 - Bot).\n")


    print("=== 3. AUTÓMATA (Validación de Cronologías / Falla en Cascada) ===")
    # Autómata Finito Determinista (AFD) para analizar secuencias de eventos técnicos en logs.
    # Alfabeto: 'O' (Operativo), 'E' (Error), 'T' (Timeout)
    # Objetivo: Reconocer si la secuencia termina en "ET" (Error seguido inmediatamente de Timeout).
    
    def acepta_patron_falla_cascada(secuencia_logs):
        state = "q0" # Inicializa el autómata en el estado inicial q0
        
        # Diccionario que define la tabla de transiciones formales del autómata (estado, símbolo) -> nuevo_estado
        transitions = {
            ("q0", "O"): "q0", 
            ("q0", "T"): "q0", 
            ("q0", "E"): "q1", # Detecta Error, avanza al estado q1
            
            ("q1", "O"): "q0", # Se recupera, vuelve al estado base q0
            ("q1", "E"): "q1", # Continúan errores, se mantiene en q1
            ("q1", "T"): "q2", # Timeout tras Error -> Llega al Estado de Aceptación (q2)
            
            ("q2", "O"): "q0", 
            ("q2", "T"): "q0", 
            ("q2", "E"): "q1",
        }
        
        # Itera sobre cada símbolo dentro de la secuencia de logs proporcionada
        for simbolo in secuencia_logs:
            # Comprueba si la transición actual es válida según la tabla definida
            if (state, simbolo) in transitions:
                state = transitions[(state, simbolo)] # Actualiza el estado actual del autómata
            else:
                return False # Retorna Falso si el símbolo no pertenece al alfabeto o no tiene transición válida
                
        # Retorna Verdadero únicamente si el autómata finaliza exactamente en el estado de aceptación q2
        return state == "q2"

    # Lista con casos de prueba cronológicos para auditar mediante el autómata
    casos_prueba = [
        "OOET",   # Termina en ET -> Falla en cascada (Aceptada)
        "OEE",    # Termina en Errores, pero sin Timeout -> (Rechazada)
        "ETOO",   # Tuvo la falla pero se estabilizó al final -> (Rechazada)
        "OOOEET"  # Múltiples errores y luego Timeout final -> (Aceptada)
    ]
    
    # Itera sobre cada caso de prueba evaluando su aceptación en el AFD e imprimiendo el dictamen
    for seq in casos_prueba:
        resultado = acepta_patron_falla_cascada(seq)
        estado = "Aceptada (Falla Crítica Confirmada)" if resultado else "Rechazada (Comportamiento Normal/Recuperado)"
        print(f"Secuencia de logs '{seq}': {estado}")

# Punto de entrada principal para ejecutar el script de forma independiente
if __name__ == "__main__":
    ejecutar_representaciones_hibridas()
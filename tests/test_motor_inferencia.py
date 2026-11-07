# tests/test_motor_inferencia.py
import unittest
from src.classifiers.motor_inferencia import Regla, inferir_con_explicacion, REGLAS_SOPORTE_TI

class TestMotorInferenciaAvanzado(unittest.TestCase):
    
    def test_caso_a_conclusion_demostrada(self):
        """Caso A: Con evidencias completas, se alcanza la conclusión final 'prioridad_critica_urgente'."""
        hechos_iniciales = {
            "imagen_procesada_ok",
            "textura_anomala_detectada",
            "hardware_critico",
            "sla_excedido"
        }
        hechos_finales, trazas, explicacion = inferir_con_explicacion(hechos_iniciales, REGLAS_SOPORTE_TI)
        
        self.assertIn("prioridad_critica_urgente", hechos_finales)
        self.assertEqual(trazas[:3], ["R1", "R2", "R3"])
        self.assertTrue(len(explicacion) > 0)

    def test_caso_b_conclusion_no_demostrada(self):
        """Caso B: Faltando la evidencia de SLA excedido, la conclusión final no se demuestra."""
        hechos_iniciales = {
            "imagen_procesada_ok",
            "textura_anomala_detectada",
            "hardware_critico"
        }
        hechos_finales, trazas, _ = inferir_con_explicacion(hechos_iniciales, REGLAS_SOPORTE_TI)
        
        self.assertNotIn("prioridad_critica_urgente", hechos_finales)
        self.assertIn("escalamiento_nivel_2", hechos_finales)
        self.assertIn("R1", trazas)
        self.assertIn("R2", trazas)
        self.assertNotIn("R3", trazas)

    def test_conservacion_hechos_iniciales(self):
        """Verifica que el conjunto de hechos iniciales no sea mutado durante la ejecución."""
        hechos_iniciales = {"imagen_procesada_ok", "textura_anomala_detectada"}
        hechos_originales = hechos_iniciales.copy()
        
        _, _, _ = inferir_con_explicacion(hechos_iniciales, REGLAS_SOPORTE_TI)
        
        self.assertEqual(hechos_iniciales, hechos_originales)

    def test_ausencia_ciclos_infinitos(self):
        """Verifica que dependencias circulares positivas no generen bucles infinitos."""
        reglas_ciclicas = [
            Regla("RC1", frozenset({"a"}), "b", "Ciclo 1"),
            Regla("RC2", frozenset({"b"}), "a", "Ciclo 2")
        ]
        hechos_iniciales = {"a"}
        
        hechos_finales, trazas, _ = inferir_con_explicacion(hechos_iniciales, reglas_ciclicas)
        
        self.assertIn("b", hechos_finales)
        self.assertEqual(trazas, ["RC1"])

if __name__ == "__main__":
    unittest.main()
"""Pruebas de la red semántica de Soporte TI (Semana 14).
Ejecutar desde la raíz del proyecto:  python -m unittest -v   (o discover -s tests)
"""
import sys
import unittest
from pathlib import Path

# Permite ejecutar el test estando en tests/ o en src/ sin configurar PYTHONPATH.
_AQUI = Path(__file__).resolve().parent
for _c in (_AQUI, _AQUI.parent, _AQUI / "src", _AQUI.parent / "src"):
    if (_c / "classifiers").is_dir() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from classifiers.red_semantica import RedSemantica, RedSemanticaSoporteTI, normalizar


def red_dominio():
    """Dominio del proyecto: 8 conceptos y 3 relaciones (es_un, requiere, se_resuelve_con)."""
    red = RedSemanticaSoporteTI()
    for oracion in (
        "servidor_bd es un hardware_critico",
        "hardware_critico es un activo_corporativo",
        "activo_corporativo requiere alimentacion_redundante",
        "estacion_trabajo es un hardware_estandar",
        "sobrecalentamiento se resuelve con limpieza_ventilacion",
    ):
        red.interpretar_oracion(oracion)
    return red


class TestNormalizacion(unittest.TestCase):
    def test_tildes_y_mayusculas(self):
        self.assertEqual(normalizar("  Hardware CRÍTICO "), "hardware_critico")

    def test_espacios_y_articulos(self):
        self.assertEqual(normalizar("El Servidor BD"), "servidor_bd")

    def test_entrada_no_texto(self):
        self.assertEqual(normalizar(None), "")
        self.assertEqual(normalizar(42), "")

    def test_alias_de_clase(self):
        self.assertIs(RedSemantica, RedSemanticaSoporteTI)


class TestInterpretacion(unittest.TestCase):
    def setUp(self):
        self.red = RedSemanticaSoporteTI()

    def test_es_un(self):
        self.assertEqual(self.red.interpretar_oracion("servidor_bd es un hardware_critico"),
                         ("servidor_bd", "es_un", "hardware_critico"))

    def test_oracion_natural_con_tildes_articulos_y_punto(self):
        t = self.red.interpretar_oracion("El Servidor BD es un Hardware Crítico.")
        self.assertEqual(t, ("servidor_bd", "es_un", "hardware_critico"))

    def test_es_una(self):
        t = self.red.interpretar_oracion("impresora es una periferico")
        self.assertEqual(t[1], "es_un")

    def test_sinonimo_necesita_se_convierte_en_requiere(self):
        t = self.red.interpretar_oracion("activo_corporativo necesita alimentacion_redundante")
        self.assertEqual(t, ("activo_corporativo", "requiere", "alimentacion_redundante"))

    def test_se_resuelve_con(self):
        t = self.red.interpretar_oracion("sobrecalentamiento se resuelve con limpieza_ventilacion")
        self.assertEqual(t[1], "se_resuelve_con")

    def test_oraciones_invalidas(self):
        casos = ["el procesador funciona muy lento hoy", "", "   ", "servidor_bd es un",
                 "es un hardware", "x es un y que requiere z", "servidor_bd es un servidor_bd",
                 "el servidor de la sala principal es un hardware critico"]
        for oracion in casos:
            with self.subTest(oracion=oracion):
                with self.assertRaises(ValueError):
                    self.red.interpretar_oracion(oracion)

    def test_entrada_no_texto_lanza_valueerror(self):
        for valor in (None, 123, ["a es un b"]):
            with self.subTest(valor=valor):
                with self.assertRaises(ValueError):
                    self.red.interpretar_oracion(valor)

    def test_oracion_invalida_no_modifica_la_red(self):
        with self.assertRaises(ValueError):
            self.red.interpretar_oracion("x es un y que requiere z")
        self.assertEqual(self.red.triples, [])
        self.assertEqual(self.red.nodos, set())

    def test_registrar_false_no_almacena(self):
        self.red.interpretar_oracion("a es un b", registrar=False)
        self.assertEqual(self.red.triples, [])

    def test_mensaje_de_error_indica_formatos(self):
        with self.assertRaises(ValueError) as ctx:
            self.red.interpretar_oracion("el equipo funciona mal")
        self.assertIn("Formatos válidos", str(ctx.exception))

    def test_relacion_no_soportada(self):
        with self.assertRaises(ValueError):
            self.red.agregar_triple("a", "vuela_hacia", "b")

    def test_triple_con_campos_vacios(self):
        with self.assertRaises(ValueError):
            self.red.agregar_triple("a", "es_un", "  ")


class TestRelacionesExplicitas(unittest.TestCase):
    def setUp(self):
        self.red = red_dominio()

    def test_triples_almacenados(self):
        self.assertIn(("servidor_bd", "es_un", "hardware_critico"), self.red.triples)
        self.assertIn(("activo_corporativo", "requiere", "alimentacion_redundante"), self.red.triples)

    def test_orden_de_insercion_y_alias_del_dashboard(self):
        self.assertEqual(self.red.triples[0], ("servidor_bd", "es_un", "hardware_critico"))
        self.assertEqual(self.red.triples, self.red.triples_explicitos)

    def test_duplicados_no_se_repiten(self):
        self.red.interpretar_oracion("servidor_bd es un hardware_critico")
        self.assertEqual(len(self.red.triples), 5)
        self.assertEqual(self.red.aristas["servidor_bd"].count(("es_un", "hardware_critico")), 1)

    def test_nodos_incluyen_conceptos_solo_destino(self):
        self.assertIn("alimentacion_redundante", self.red.nodos)
        self.assertEqual(len(self.red.nodos), 8)

    def test_conceptos_relacionados(self):
        rel = self.red.conceptos_relacionados("hardware_critico")
        self.assertEqual(rel["salientes"], [("es_un", "activo_corporativo")])
        self.assertEqual(rel["entrantes"], [("servidor_bd", "es_un")])

    def test_subtipos(self):
        self.assertEqual(self.red.subtipos("activo_corporativo"), ["hardware_critico", "servidor_bd"])
        self.assertEqual(self.red.subtipos("servidor_bd"), [])


class TestInferenciaJerarquia(unittest.TestCase):
    def setUp(self):
        self.red = red_dominio()

    def test_transitividad_dos_pasos(self):
        self.assertTrue(self.red.es_un("servidor_bd", "activo_corporativo"))
        self.assertNotIn(("servidor_bd", "es_un", "activo_corporativo"), self.red.triples)  # no se almacenó

    def test_transitividad_tres_pasos(self):
        self.red.interpretar_oracion("activo_corporativo es un recurso_empresarial")
        self.assertTrue(self.red.es_un("servidor_bd", "recurso_empresarial"))

    def test_sin_evidencia(self):
        self.assertFalse(self.red.es_un("estacion_trabajo", "hardware_critico"))
        self.assertFalse(self.red.es_un("estacion_trabajo", "activo_corporativo"))

    def test_la_jerarquia_no_es_simetrica(self):
        self.assertFalse(self.red.es_un("activo_corporativo", "servidor_bd"))

    def test_relaciones_distintas_de_es_un_no_forman_jerarquia(self):
        self.assertFalse(self.red.es_un("sobrecalentamiento", "limpieza_ventilacion"))

    def test_concepto_inexistente(self):
        self.assertFalse(self.red.es_un("impresora_3d", "activo_corporativo"))
        self.assertFalse(self.red.es_un("impresora_3d", "impresora_3d"))

    def test_identidad_solo_para_conceptos_existentes(self):
        self.assertTrue(self.red.es_un("servidor_bd", "servidor_bd"))

    def test_consulta_normaliza_la_entrada(self):
        self.assertTrue(self.red.es_un("Servidor BD", "Activo Corporativo"))

    def test_ciclos_no_provocan_bucle_infinito(self):
        red = RedSemanticaSoporteTI()
        red.agregar_triple("a", "es_un", "b")
        red.agregar_triple("b", "es_un", "a")
        red.agregar_triple("c", "es_un", "d")
        self.assertTrue(red.es_un("a", "b"))
        self.assertFalse(red.es_un("a", "d"))
        self.assertEqual(red.subtipos("a"), ["b"])

    def test_explicar_devuelve_la_cadena_de_triples(self):
        self.assertEqual(self.red.explicar_es_un("servidor_bd", "activo_corporativo"),
                         [("servidor_bd", "es_un", "hardware_critico"),
                          ("hardware_critico", "es_un", "activo_corporativo")])
        self.assertIsNone(self.red.explicar_es_un("estacion_trabajo", "hardware_critico"))

    def test_justificaciones(self):
        j = self.red.justificar_es_un
        self.assertIn("Hecho explícito", j("servidor_bd", "hardware_critico"))
        self.assertIn("transitividad", j("servidor_bd", "activo_corporativo"))
        self.assertIn("Sin evidencia", j("estacion_trabajo", "hardware_critico"))
        self.assertIn("no existe en la red", j("fantasma", "hardware_critico"))


class TestHerencia(unittest.TestCase):
    def setUp(self):
        self.red = red_dominio()

    def test_herencia_dos_niveles(self):
        self.assertIn("alimentacion_redundante", self.red.consultar_propiedades("servidor_bd"))

    def test_herencia_un_nivel(self):
        self.assertEqual(self.red.consultar_propiedades("hardware_critico"), ["alimentacion_redundante"])

    def test_sin_evidencia_no_hereda(self):
        self.assertEqual(self.red.consultar_propiedades("estacion_trabajo"), [])
        self.assertEqual(self.red.consultar_propiedades("concepto_inexistente"), [])

    def test_no_hereda_hacia_abajo(self):
        self.red.interpretar_oracion("servidor_bd requiere respaldo_diario")
        self.assertEqual(self.red.consultar_propiedades("servidor_bd"),
                         ["alimentacion_redundante", "respaldo_diario"])
        self.assertEqual(self.red.consultar_propiedades("activo_corporativo"), ["alimentacion_redundante"])

    def test_solo_se_heredan_relaciones_heredables(self):
        self.red.agregar_triple("hardware_critico", "se_resuelve_con", "mantenimiento_preventivo")
        self.assertNotIn("mantenimiento_preventivo", self.red.consultar_propiedades("servidor_bd"))
        with self.assertRaises(ValueError):
            self.red.propiedades_con_origen("servidor_bd", "se_resuelve_con")

    def test_origen_de_la_propiedad_heredada(self):
        info = self.red.propiedades_con_origen("servidor_bd")[0]
        self.assertEqual(info["declarada_en"], "activo_corporativo")
        self.assertEqual(len(info["ruta"]), 2)

    def test_propiedad_propia_no_tiene_ruta(self):
        self.red.interpretar_oracion("servidor_bd requiere respaldo_diario")
        propias = {d["propiedad"]: d for d in self.red.propiedades_con_origen("servidor_bd")}
        self.assertEqual(propias["respaldo_diario"]["ruta"], [])

    def test_justificar_propiedades(self):
        texto = " ".join(self.red.justificar_propiedades("servidor_bd"))
        self.assertIn("heredada (R2)", texto)
        self.assertIn("servidor_bd -> hardware_critico -> activo_corporativo", texto)


if __name__ == "__main__":
    unittest.main()
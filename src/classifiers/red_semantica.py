"""
Red semántica del Asistente de Soporte TI Híbrido (Semana 14).

Modelo
------
* Concepto  -> nodo de la red (ej. servidor_bd).
* Relación  -> arista dirigida y etiquetada (ver RELACIONES).
* Triple    -> (sujeto, relación, objeto); es la unidad mínima de conocimiento.

Reglas de inferencia
--------------------
R1 (transitividad): si A es_un B  y  B es_un C  =>  A es_un C.
R2 (herencia):      si A es_un B  y  B requiere P  =>  A requiere P.

Supuesto de mundo cerrado: solo se concluye lo que se puede demostrar con los
triples registrados. Un resultado False significa "sin evidencia", no "falso".
"""
import re
import unicodedata
from collections import deque

# ---------------------------------------------------------------------------
# Vocabulario controlado: cada relación tiene un significado explícito.
# ---------------------------------------------------------------------------
RELACIONES = {
    "es_un": "El sujeto es una subcategoría o instancia del objeto. Es transitiva.",
    "requiere": "El sujeto necesita el objeto para operar. Se hereda por la jerarquía es_un.",
    "se_resuelve_con": "El sujeto se corrige aplicando el objeto (procedimiento o acción).",
}
RELACIONES_HEREDABLES = {"requiere"}
ALIAS_RELACION = {"necesita": "requiere"}

_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
_PALABRAS_RELACION = {"es", "requiere", "necesita", "resuelve"}
MAX_PALABRAS_CONCEPTO = 3

_PATRONES = (
    (re.compile(r"^(?P<s>.+?) es (?:un|una) (?P<o>.+)$"), "es_un"),
    (re.compile(r"^(?P<s>.+?) (?:requiere|necesita) (?P<o>.+)$"), "requiere"),
    (re.compile(r"^(?P<s>.+?) se resuelve con (?P<o>.+)$"), "se_resuelve_con"),
)
FORMATOS_ADMITIDOS = ("'<A> es un <B>', '<A> requiere <B>' (o 'necesita'), "
                      "'<A> se resuelve con <B>'")


def _sin_tildes(texto: str) -> str:
    base = unicodedata.normalize("NFD", texto.casefold().strip())
    return "".join(c for c in base if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    """Forma canónica de un concepto: minúsculas, sin tildes, sin artículo
    inicial y con '_' entre palabras. 'El Servidor BD' -> 'servidor_bd'.
    Limitación: 'ñ' pasa a 'n' (año y ano quedan iguales)."""
    if not isinstance(texto, str):
        return ""
    palabras = [p for p in re.split(r"[^a-z0-9]+", _sin_tildes(texto)) if p]
    while len(palabras) > 1 and palabras[0] in _ARTICULOS:
        palabras.pop(0)
    return "_".join(palabras)


def _limpiar_oracion(texto: str) -> str:
    base = re.sub(r"[.!?;,]+$", "", _sin_tildes(texto))
    return re.sub(r"\s+", " ", base).strip()


def _concepto_desde_oracion(texto: str) -> str:
    """Valida un fragmento de oración como concepto simple y lo normaliza."""
    palabras = texto.split()
    if len(palabras) > 1 and palabras[0] in _ARTICULOS:
        palabras = palabras[1:]
    if not palabras:
        raise ValueError("concepto vacío")
    if len(palabras) > MAX_PALABRAS_CONCEPTO:
        raise ValueError(f"concepto demasiado largo '{texto}' (use guion_bajo para unir palabras)")
    if any(p in _PALABRAS_RELACION for p in palabras):
        raise ValueError(f"'{texto}' mezcla un concepto con otra relación")
    return normalizar(texto)


class RedSemanticaSoporteTI:
    """Red semántica con triples explícitos e inferencia por jerarquía y herencia."""

    def __init__(self):
        self.aristas = {}      # sujeto -> [(relacion, objeto)]
        self.nodos = set()     # todos los conceptos (sujetos y objetos)
        self._triples = {}     # dict como conjunto ordenado (orden de inserción)

    # ----------------------------- Almacenamiento -----------------------------
    @property
    def triples(self) -> list:
        return list(self._triples)

    @property
    def triples_explicitos(self) -> list:  # nombre usado por el dashboard
        return self.triples

    def agregar_triple(self, sujeto: str, relacion: str, objeto: str) -> tuple:
        """Valida, normaliza y almacena un triple. Devuelve el triple guardado."""
        s, o = normalizar(sujeto), normalizar(objeto)
        r = normalizar(relacion)
        r = ALIAS_RELACION.get(r, r)
        if not s or not r or not o:
            raise ValueError("El sujeto, la relación y el objeto no pueden estar vacíos.")
        if r not in RELACIONES:
            raise ValueError(f"Relación no soportada: '{r}'. Válidas: {sorted(RELACIONES)}")
        if s == o:
            raise ValueError(f"Un concepto no puede relacionarse consigo mismo: '{s}'.")
        triple = (s, r, o)
        if triple not in self._triples:
            self._triples[triple] = None
            self.aristas.setdefault(s, []).append((r, o))
        self.nodos.update((s, o))
        return triple

    def interpretar_oracion(self, oracion: str, registrar: bool = True) -> tuple:
        """Convierte una oración simple en un triple (sujeto, relación, objeto).
        Lanza ValueError si la oración no tiene un formato admitido."""
        if not isinstance(oracion, str) or not oracion.strip():
            raise ValueError("La oración debe ser un texto no vacío.")
        limpia = _limpiar_oracion(oracion)
        motivo = None
        for patron, relacion in _PATRONES:
            m = patron.match(limpia)
            if not m:
                continue
            try:
                sujeto = _concepto_desde_oracion(m.group("s"))
                objeto = _concepto_desde_oracion(m.group("o"))
            except ValueError as error:
                motivo = str(error)
                continue
            if registrar:
                return self.agregar_triple(sujeto, relacion, objeto)
            return (sujeto, relacion, objeto)
        detalle = f" Motivo: {motivo}." if motivo else ""
        raise ValueError(f"Oración no admitida: '{oracion.strip()}'.{detalle} Formatos válidos: {FORMATOS_ADMITIDOS}.")

    # ------------------------------- Consultas --------------------------------
    def conceptos_relacionados(self, concepto: str) -> dict:
        """Relaciones directas de un concepto: las que salen y las que llegan."""
        c = normalizar(concepto)
        entrantes = [(s, r) for s, lista in self.aristas.items() for r, o in lista if o == c]
        return {"salientes": list(self.aristas.get(c, [])), "entrantes": entrantes}

    def subtipos(self, categoria: str) -> list:
        """Todos los conceptos que son_un categoría (directa o transitivamente)."""
        raiz = normalizar(categoria)
        hijos = {}
        for s, lista in self.aristas.items():
            for r, o in lista:
                if r == "es_un":
                    hijos.setdefault(o, []).append(s)
        vistos, cola = {raiz}, deque([raiz])
        while cola:
            for h in hijos.get(cola.popleft(), []):
                if h not in vistos:
                    vistos.add(h)
                    cola.append(h)
        return sorted(vistos - {raiz})

    # ------------------------------- Inferencia -------------------------------
    def _ancestros_es_un(self, concepto: str) -> dict:
        """BFS por es_un. Devuelve {nodo: padre}; 'visitados' = claves (evita ciclos)."""
        padres, cola = {concepto: None}, deque([concepto])
        while cola:
            actual = cola.popleft()
            for rel, dest in self.aristas.get(actual, []):
                if rel == "es_un" and dest not in padres:
                    padres[dest] = actual
                    cola.append(dest)
        return padres

    @staticmethod
    def _ruta(padres: dict, nodo: str) -> list:
        ruta = []
        while padres[nodo] is not None:
            ruta.append((padres[nodo], "es_un", nodo))
            nodo = padres[nodo]
        return ruta[::-1]

    def explicar_es_un(self, origen: str, categoria: str):
        """Prueba de 'origen es_un categoria': lista de triples es_un encadenados,
        [] si es identidad, o None si no hay evidencia."""
        org, cat = normalizar(origen), normalizar(categoria)
        if org not in self.nodos or cat not in self.nodos:
            return None
        if org == cat:
            return []
        padres = self._ancestros_es_un(org)
        return self._ruta(padres, cat) if cat in padres else None

    def es_un(self, origen: str, categoria: str) -> bool:
        return self.explicar_es_un(origen, categoria) is not None

    def justificar_es_un(self, origen: str, categoria: str) -> str:
        """Texto que explica cómo se obtuvo (o por qué no) la conclusión."""
        org, cat = normalizar(origen), normalizar(categoria)
        for c in (org, cat):
            if c not in self.nodos:
                return f"Sin evidencia: el concepto '{c}' no existe en la red."
        camino = self.explicar_es_un(org, cat)
        if camino is None:
            return (f"Sin evidencia: no existe una cadena es_un de '{org}' a '{cat}' "
                    f"en los hechos registrados.")
        if not camino:
            return f"Identidad: '{org}' es el mismo concepto."
        if len(camino) == 1:
            return f"Hecho explícito: ({org}, es_un, {cat}) está almacenado directamente."
        pasos = " + ".join(f"({s} es_un {o})" for s, _, o in camino)
        return f"Inferido por transitividad (R1): {pasos} => ({org} es_un {cat})"

    def propiedades_con_origen(self, concepto: str, relacion: str = "requiere") -> list:
        """Propiedades propias y heredadas, indicando dónde se declararon y la
        ruta es_un que permite heredarlas."""
        if relacion not in RELACIONES_HEREDABLES:
            raise ValueError(f"La relación '{relacion}' no es heredable. Heredables: {sorted(RELACIONES_HEREDABLES)}")
        c = normalizar(concepto)
        if c not in self.nodos:
            return []
        padres = self._ancestros_es_un(c)
        origen = {}
        for ancestro in padres:  # orden BFS: primero el más cercano
            for rel, dest in self.aristas.get(ancestro, []):
                if rel == relacion:
                    origen.setdefault(dest, ancestro)
        return [{"propiedad": p, "declarada_en": a, "ruta": self._ruta(padres, a)}
                for p, a in sorted(origen.items())]

    def consultar_propiedades(self, concepto: str) -> list:
        return [d["propiedad"] for d in self.propiedades_con_origen(concepto)]

    def justificar_propiedades(self, concepto: str) -> list:
        """Una línea explicativa por cada propiedad (propia o heredada)."""
        lineas = []
        for d in self.propiedades_con_origen(concepto):
            if not d["ruta"]:
                lineas.append(f"{d['propiedad']}: declarada directamente en '{d['declarada_en']}'.")
            else:
                cadena = " -> ".join([d["ruta"][0][0]] + [t[2] for t in d["ruta"]])
                lineas.append(f"{d['propiedad']}: heredada (R2) de '{d['declarada_en']}' por la cadena {cadena}.")
        return lineas

    def resumen(self) -> str:
        return f"{len(self.nodos)} conceptos y {len(self._triples)} triples explícitos"


RedSemantica = RedSemanticaSoporteTI  # alias de compatibilidad con la práctica de clase
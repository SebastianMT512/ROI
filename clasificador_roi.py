"""
clasificador_roi.py
===================
Aplicación interactiva para clasificación de imágenes multiespectrales
Landsat 8 mediante polígonos ROI (Región de Interés).

Desarrollado para el taller universitario de Procesamiento de Imágenes
Universidad Distrital Francisco José de Caldas — Maestría

Uso:
    python clasificador_roi.py

Controles:
    - Clic izquierdo: agrega vértice al polígono en curso
    - Doble clic:     cierra el polígono
    - Botón [Nuevo Polígono]:  inicia un nuevo polígono
    - Botón [Limpiar Último]:  elimina el último polígono dibujado
    - Botón [Limpiar Todo]:    elimina todos los polígonos
    - Botón [Exportar CSV]:    guarda la matriz de características
    - Botón [Mostrar Tabla]:   imprime la tabla en consola

Dependencias:
    rasterio, numpy, matplotlib, pandas, shapely
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — Ajusta la ruta de la imagen aquí
# ─────────────────────────────────────────────────────────────────────────────
# Coloca el archivo .tif en la misma carpeta que este script
# o cambia esta ruta a la ubicación de tu imagen
RUTA_IMAGEN = r"LC08_L1TP_009058_20251208_20251217_02_T1_Bandas1-7.tif"

# Recorte de interés (filas y columnas en coordenadas de la imagen completa)
FILA_INICIO  = 3163
FILA_FIN     = 4531
COL_INICIO   = 2285
COL_FIN      = 5236

# Tamaño de pixel Landsat 8 en metros
TAMANO_PIXEL_M = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # Backend interactivo; cambia a "Qt5Agg" si prefieres
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons, Button
from matplotlib.patches import Polygon as MplPolygon

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("ERROR: 'rasterio' no está instalado. Ejecuta:")
    print("  pip install rasterio --break-system-packages")
    sys.exit(1)

try:
    from shapely.geometry import Polygon as ShapelyPolygon, Point
except ImportError:
    print("ERROR: 'shapely' no está instalado. Ejecuta:")
    print("  pip install shapely --break-system-packages")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIÓN DE CLASES Y COLORES
# ─────────────────────────────────────────────────────────────────────────────

CLASES = {
    "Agua":              (0.10, 0.28, 0.45),   # azul-teal oscuro (lago izquierdo)
    "Cultivos":          (0.60, 0.75, 0.30),   # verde claro / amarillo-verde (valle)
    "Zona Urbana":       (0.62, 0.57, 0.50),   # gris-beige (zonas grises del mapa)
    "Vegetacion/Bosque": (0.10, 0.32, 0.12),   # verde oscuro (laderas densas)
    "Paramo":            (0.35, 0.20, 0.10),   # marrón oscuro (derecha del mapa)
    "Nieve/Hielo":       (0.95, 0.95, 0.95),   # blanco casi puro (Nevado del Huila)
    "Suelo":             (0.55, 0.38, 0.18),   # marrón medio (suelo desnudo)
}
NOMBRES_BANDAS = ["B1_mean", "B2_mean", "B3_mean", "B4_mean",
                  "B5_mean", "B6_mean", "B7_mean"]

COLUMNAS_TABLA = ["ID", "Clase", "Area_px", "Area_km2"] + NOMBRES_BANDAS

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE CARGA Y NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def cargar_imagen(ruta: str) -> tuple:
    """
    Carga el GeoTIFF multibanda y aplica el recorte configurado.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo GeoTIFF.

    Retorna
    -------
    bandas : np.ndarray  — shape (7, filas, cols), dtype float32
    perfil : dict        — metadatos rasterio del recorte
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró la imagen en:\n  {ruta}\n"
            "Verifica la variable RUTA_IMAGEN al inicio del script."
        )

    filas = FILA_FIN - FILA_INICIO
    cols  = COL_FIN  - COL_INICIO

    # Ventana de lectura (col_off, row_off, width, height)
    ventana = Window(col_off=COL_INICIO, row_off=FILA_INICIO,
                     width=cols, height=filas)

    with rasterio.open(ruta) as src:
        print(f"[INFO] Imagen: {src.width}×{src.height} px, {src.count} bandas")
        print(f"[INFO] CRS: {src.crs}")
        print(f"[INFO] Recorte: filas {FILA_INICIO}–{FILA_FIN}, "
              f"cols {COL_INICIO}–{COL_FIN} → {filas}×{cols} px")

        # Leer todas las bandas en el recorte
        bandas = src.read(window=ventana).astype(np.float32)

        # Actualizar perfil para el recorte
        perfil = src.profile.copy()
        perfil.update(width=cols, height=filas,
                      transform=src.window_transform(ventana))

    print(f"[INFO] Datos cargados: shape={bandas.shape}, "
          f"dtype={bandas.dtype}")
    return bandas, perfil


def normalizar_banda(banda: np.ndarray,
                     p_low: float = 2.0,
                     p_high: float = 98.0) -> np.ndarray:
    """
    Normaliza una banda al rango [0, 1] usando percentiles para
    mejorar el contraste visual (estiramiento lineal de histograma).

    Parámetros
    ----------
    banda  : np.ndarray  — banda 2D (filas × cols)
    p_low  : float       — percentil inferior (por defecto 2)
    p_high : float       — percentil superior (por defecto 98)

    Retorna
    -------
    np.ndarray  — banda normalizada en [0, 1]
    """
    # Ignorar ceros (píxeles sin datos) para el cálculo de percentiles
    validos = banda[banda > 0]
    if validos.size == 0:
        return np.zeros_like(banda)

    vmin = np.percentile(validos, p_low)
    vmax = np.percentile(validos, p_high)

    if vmax == vmin:
        return np.zeros_like(banda)

    norm = (banda - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0)


def crear_composicion_rgb(bandas: np.ndarray) -> np.ndarray:
    """
    Crea composición en verdadero color (R:B4 – G:B3 – B:B2).

    Esta composición resalta:
      - Vegetación activa:  tonos verdes oscuros
      - Cuerpos de agua:    azul-teal oscuro
      - Zonas de páramo:    marrón oscuro
      - Nieve/glaciares:    blanco
      - Suelo desnudo:      marrón medio/beige

    Parámetros
    ----------
    bandas : np.ndarray  — shape (7, filas, cols)

    Retorna
    -------
    rgb : np.ndarray  — shape (filas, cols, 3), valores en [0,1]
    """
    # Índices: B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6
    r = normalizar_banda(bandas[3])  # B4 - Rojo
    g = normalizar_banda(bandas[2])  # B3 - Verde
    b = normalizar_banda(bandas[1])  # B2 - Azul

    rgb = np.dstack([r, g, b])
    return rgb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE ANÁLISIS DE POLÍGONO
# ─────────────────────────────────────────────────────────────────────────────

def analizar_poligono(vertices: list, bandas: np.ndarray) -> dict:
    """
    Calcula el área y el vector de características (media por banda)
    de los píxeles contenidos dentro de un polígono.

    Algoritmo:
      1. Construye el polígono con Shapely
      2. Determina el bounding-box para limitar la búsqueda
      3. Verifica cada centro de píxel dentro del bounding-box
         usando el método contains() de Shapely (ray-casting interno)
      4. Extrae los valores de reflectancia de los píxeles dentro
      5. Calcula la media por banda

    Parámetros
    ----------
    vertices : list       — lista de (col, fila) en coordenadas de imagen
    bandas   : np.ndarray — shape (7, filas, cols)

    Retorna
    -------
    dict con claves:
      area_px  (int)    — número de píxeles dentro del polígono
      area_km2 (float)  — área en km²
      medias   (list)   — media de reflectancia por banda [B1..B7]
    """
    if len(vertices) < 3:
        return None

    poligono = ShapelyPolygon(vertices)

    # Reparar geometría inválida (auto-intersecciones, etc.)
    if not poligono.is_valid:
        poligono = poligono.buffer(0)

    # Bounding box del polígono para limitar la búsqueda de píxeles
    minx, miny, maxx, maxy = poligono.bounds
    col_min  = max(0, int(np.floor(minx)))
    col_max  = min(bandas.shape[2] - 1, int(np.ceil(maxx)))
    fila_min = max(0, int(np.floor(miny)))
    fila_max = min(bandas.shape[1] - 1, int(np.ceil(maxy)))

    # Crear grilla de centros de píxel dentro del bounding-box
    cols_rango  = np.arange(col_min,  col_max  + 1)
    filas_rango = np.arange(fila_min, fila_max + 1)
    cols_grid, filas_grid = np.meshgrid(cols_rango, filas_rango)
    coords = np.column_stack([cols_grid.ravel(), filas_grid.ravel()])

    # Punto-en-polígono: verificar cada píxel con Shapely
    mascara = np.array([
        poligono.contains(Point(float(c), float(f)))
        for c, f in coords
    ], dtype=bool)

    # Fallback: si ningún píxel cae exactamente dentro,
    # incluir los que estén a menos de medio píxel del borde
    if not mascara.any():
        mascara = np.array([
            poligono.distance(Point(float(c), float(f))) < 0.5
            for c, f in coords
        ], dtype=bool)

    if not mascara.any():
        return {"area_px": 0, "area_km2": 0.0, "medias": [0.0] * 7}

    # Extraer índices de fila y columna de los píxeles dentro
    cols_dentro  = coords[mascara, 0].astype(int)
    filas_dentro = coords[mascara, 1].astype(int)

    area_px  = int(mascara.sum())
    # Conversión: píxel 30m × 30m = 900 m² = 0.0009 km²
    area_km2 = area_px * (TAMANO_PIXEL_M ** 2) / 1_000_000.0

    # Calcular media espectral por banda
    medias = []
    for b in range(7):
        valores = bandas[b, filas_dentro, cols_dentro]
        medias.append(float(np.mean(valores)))

    return {"area_px": area_px, "area_km2": area_km2, "medias": medias}


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL — Aplicación interactiva
# ─────────────────────────────────────────────────────────────────────────────

class ClasificadorROI:
    """
    Aplicación interactiva de clasificación supervisada por ROI.

    La interfaz se compone de:
      - Eje principal:     imagen Landsat con polígonos dibujados encima
      - Panel izquierdo:   RadioButtons para selección de clase
      - Panel inferior:    botones de control y barra de estado
      - Leyenda de colores de clases

    Flujo de trabajo típico:
      1. Presionar [Nuevo Polígono]
      2. Clic izquierdo para agregar vértices
      3. Doble clic para cerrar y analizar
      4. Seleccionar la clase en el panel izquierdo
      5. Repetir para todos los ROIs de interés
      6. [Exportar CSV] para guardar los vectores característicos
    """

    def __init__(self, bandas: np.ndarray, perfil: dict, ruta_imagen: str):
        """
        Inicializa la aplicación.

        Parámetros
        ----------
        bandas       : np.ndarray — datos espectrales (7, filas, cols)
        perfil       : dict       — metadatos rasterio
        ruta_imagen  : str        — ruta del archivo para mostrar en título
        """
        self.bandas      = bandas
        self.perfil      = perfil
        self.ruta_imagen = ruta_imagen

        # ── Estado del dibujo ────────────────────────────────────────────────
        self.dibujando          = False  # True cuando hay polígono en curso
        self.vertices_actuales  = []     # Vértices del polígono activo [(col,fila),...]
        self.linea_provisional  = None   # Line2D temporal que sigue el cursor
        self.artistas_prov      = []     # Artistas provisionales del polígono en curso

        # ── Historial de polígonos completados ───────────────────────────────
        # Cada elemento: {id, vertices, clase, patch, texto, analisis}
        self.poligonos = []

        # ── Tabla acumulativa de vectores característicos ────────────────────
        self.tabla = pd.DataFrame(columns=COLUMNAS_TABLA)

        # ── Clase actualmente seleccionada ───────────────────────────────────
        self.clase_actual = list(CLASES.keys())[0]

        # Construir interfaz y conectar eventos
        self._construir_interfaz()
        self._conectar_eventos()

        print("\n[LISTO] Aplicación iniciada.")
        print("  → Clic izquierdo: agregar vértice")
        print("  → Doble clic:     cerrar polígono")
        print("  → Presiona 'Nuevo Polígono' para empezar\n")

    # ────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ────────────────────────────────────────────────────────────────────────

    def _construir_interfaz(self):
        """
        Crea la figura de matplotlib con todos los ejes y widgets.

        Layout de la figura (proporciones aproximadas):
          ┌──────────────────────────────────────────────┐
          │ [Radio]  │         Imagen Landsat             │
          │ Clases   │    (con polígonos encima)          │
          │          │                                    │
          ├──────────┴────────────────────────────────────┤
          │           Barra de estado                     │
          ├───────────────────────────────────────────────┤
          │  [Nuevo] [Limpiar] [Todo] [Export] [Tabla]    │
          └───────────────────────────────────────────────┘
        """
        self.fig = plt.figure(
            figsize=(16, 10),
            facecolor="#1a1a2e",
            num="Clasificador ROI — Landsat 8"
        )
        self.fig.patch.set_facecolor("#1a1a2e")

        # ── Eje principal: imagen ────────────────────────────────────────────
        # [left, bottom, width, height] en fracción de la figura
        self.ax_img = self.fig.add_axes([0.18, 0.18, 0.79, 0.76])

        # ── Panel selector de clase (RadioButtons) ───────────────────────────
        self.ax_radio = self.fig.add_axes([0.01, 0.35, 0.16, 0.42],
                                           facecolor="#0f3460")

        # ── Botones de control ───────────────────────────────────────────────
        self.ax_btn_nuevo   = self.fig.add_axes([0.18, 0.04, 0.12, 0.065])
        self.ax_btn_limpiar = self.fig.add_axes([0.31, 0.04, 0.12, 0.065])
        self.ax_btn_todo    = self.fig.add_axes([0.44, 0.04, 0.12, 0.065])
        self.ax_btn_export  = self.fig.add_axes([0.57, 0.04, 0.12, 0.065])
        self.ax_btn_tabla   = self.fig.add_axes([0.70, 0.04, 0.12, 0.065])

        # ── Mostrar imagen (Verdadero Color B4-B3-B2) ────────────────────────────
        self.rgb = crear_composicion_rgb(self.bandas)
        self.im  = self.ax_img.imshow(
            self.rgb, origin="upper", interpolation="bilinear"
        )

        # Estilo visual del eje imagen
        self.ax_img.set_facecolor("#000000")
        self.ax_img.tick_params(colors="#888888", labelsize=7)
        for spine in self.ax_img.spines.values():
            spine.set_edgecolor("#444444")

        # Título informativo
        nombre_archivo = os.path.basename(self.ruta_imagen)
        self.ax_img.set_title(
            f"Landsat 8 — Verdadero Color B4-B3-B2  |  {nombre_archivo}\n"
            f"Recorte: filas {FILA_INICIO}–{FILA_FIN}, "
            f"cols {COL_INICIO}–{COL_FIN}  "
            f"({self.bandas.shape[1]} × {self.bandas.shape[2]} px)",
            color="white", fontsize=9, pad=8
        )

        # Texto de instrucciones en la esquina inferior izquierda de la imagen
        self.ax_img.text(
            0.01, 0.015,
            "Clic izq → vértice  |  Doble clic → cerrar polígono",
            transform=self.ax_img.transAxes,
            color="yellow", fontsize=8, alpha=0.9,
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3)
        )

        # ── RadioButtons de selección de clase ───────────────────────────────
        nombres_clases = list(CLASES.keys())
        self.radio = RadioButtons(
            self.ax_radio, nombres_clases, activecolor="#ffffff"
        )

        # Colorear etiquetas con el color de cada clase
        for label, nombre in zip(self.radio.labels, nombres_clases):
            label.set_color(CLASES[nombre])
            label.set_fontsize(8)

        self.ax_radio.set_title("Clase ROI", color="white",
                                 fontsize=9, pad=6)
        self.radio.on_clicked(self._cambiar_clase)

        # ── Leyenda de colores en la imagen ──────────────────────────────────
        parches_leyenda = [
            mpatches.Patch(color=color, label=nombre)
            for nombre, color in CLASES.items()
        ]
        self.ax_img.legend(
            handles=parches_leyenda,
            loc="upper right",
            fontsize=7,
            framealpha=0.75,
            facecolor="#0f3460",
            edgecolor="#444",
            labelcolor="white",
            title="Clases",
            title_fontsize=8
        )

        # ── Botones de control ───────────────────────────────────────────────
        estilo_btn = dict(color="#16213e", hovercolor="#0f3460")

        self.btn_nuevo   = Button(self.ax_btn_nuevo,
                                   "Nuevo\nPolígono",  **estilo_btn)
        self.btn_limpiar = Button(self.ax_btn_limpiar,
                                   "Limpiar\nÚltimo",  **estilo_btn)
        self.btn_todo    = Button(self.ax_btn_todo,
                                   "Limpiar\nTodo",     **estilo_btn)
        self.btn_export  = Button(self.ax_btn_export,
                                   "Exportar\nCSV",     **estilo_btn)
        self.btn_tabla   = Button(self.ax_btn_tabla,
                                   "Mostrar\nTabla",    **estilo_btn)

        for btn in [self.btn_nuevo, self.btn_limpiar, self.btn_todo,
                    self.btn_export, self.btn_tabla]:
            btn.label.set_color("white")
            btn.label.set_fontsize(8)

        # Conectar callbacks de botones
        self.btn_nuevo.on_clicked(self._nuevo_poligono)
        self.btn_limpiar.on_clicked(self._limpiar_ultimo)
        self.btn_todo.on_clicked(self._limpiar_todo)
        self.btn_export.on_clicked(self._exportar_csv)
        self.btn_tabla.on_clicked(self._mostrar_tabla)

        # ── Barra de estado (texto informativo) ──────────────────────────────
        self.ax_status = self.fig.add_axes([0.18, 0.12, 0.79, 0.045])
        self.ax_status.set_facecolor("#0f3460")
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        for spine in self.ax_status.spines.values():
            spine.set_edgecolor("#444")

        self.texto_status = self.ax_status.text(
            0.01, 0.50,
            "Presiona [Nuevo Polígono] para comenzar a dibujar.",
            transform=self.ax_status.transAxes,
            color="lightgreen", fontsize=8.5, va="center"
        )

        # Ajustar el layout
        self.fig.text(
            0.095, 0.30,
            "Selecciona\nla clase\nantes de\ncerrar el\npolígono",
            ha="center", va="center",
            color="#aaaaaa", fontsize=7.5
        )

    # ────────────────────────────────────────────────────────────────────────
    # CONEXIÓN DE EVENTOS
    # ────────────────────────────────────────────────────────────────────────

    def _conectar_eventos(self):
        """Registra los callbacks de eventos de ratón."""
        self.fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    # ────────────────────────────────────────────────────────────────────────
    # CALLBACKS DE RATÓN
    # ────────────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        """
        Maneja los clics del ratón sobre el eje de imagen.

        Clic simple  → agrega un vértice al polígono en curso
        Doble clic   → cierra y analiza el polígono
        """
        # Solo procesar si estamos dibujando y el clic es en la imagen
        if not self.dibujando:
            return
        if event.inaxes != self.ax_img:
            return
        if event.button != 1:   # Solo botón izquierdo
            return

        col  = event.xdata
        fila = event.ydata

        # Verificar que las coordenadas son válidas
        if col is None or fila is None:
            return

        if event.dblclick:
            # Doble clic: cerrar el polígono si tiene suficientes vértices
            if len(self.vertices_actuales) >= 3:
                self._cerrar_poligono()
            else:
                self._actualizar_status(
                    "⚠  Necesitas al menos 3 vértices para cerrar el polígono.",
                    color="orange"
                )
        else:
            # Clic simple: agregar vértice
            self.vertices_actuales.append((col, fila))
            self._redibujar_provisional()
            n = len(self.vertices_actuales)
            self._actualizar_status(
                f"Vértice {n} agregado en ({col:.0f}, {fila:.0f}).  "
                "Doble clic para cerrar el polígono.",
                color="lightyellow"
            )

    def _on_move(self, event):
        """
        Dibuja una línea provisional desde el último vértice
        hasta la posición actual del cursor para guiar al usuario.
        """
        if not self.dibujando:
            return
        if event.inaxes != self.ax_img:
            return
        if len(self.vertices_actuales) == 0:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Eliminar línea anterior al cursor
        if self.linea_provisional is not None:
            try:
                self.linea_provisional.remove()
            except ValueError:
                pass
            self.linea_provisional = None

        # Dibujar nueva línea provisional hacia el cursor
        ultimo_col, ultima_fila = self.vertices_actuales[-1]
        self.linea_provisional, = self.ax_img.plot(
            [ultimo_col, event.xdata],
            [ultima_fila, event.ydata],
            "--", color="white", linewidth=1.0, alpha=0.65, zorder=10
        )
        self.fig.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────────
    # LÓGICA DE DIBUJO
    # ────────────────────────────────────────────────────────────────────────

    def _redibujar_provisional(self):
        """
        Redibuja los puntos y líneas provisionales del polígono en curso.
        Elimina los artistas anteriores para evitar duplicados.
        """
        # Limpiar artistas provisionales anteriores
        for a in self.artistas_prov:
            try:
                a.remove()
            except ValueError:
                pass
        self.artistas_prov.clear()

        if len(self.vertices_actuales) < 1:
            return

        cols  = [v[0] for v in self.vertices_actuales]
        filas = [v[1] for v in self.vertices_actuales]
        color = CLASES[self.clase_actual]

        # Línea que une los vértices (abierta)
        linea, = self.ax_img.plot(
            cols, filas, "-o",
            color=color,
            linewidth=1.8,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=9
        )
        self.artistas_prov.append(linea)

        self.fig.canvas.draw_idle()

    def _cerrar_poligono(self):
        """
        Cierra el polígono activo:
          1. Elimina artistas provisionales
          2. Dibuja el polígono relleno definitivo
          3. Agrega etiqueta con el número de polígono
          4. Llama al análisis espectral
          5. Actualiza la tabla de características
          6. Reinicia el estado de dibujo
        """
        # ── Limpiar artistas provisionales ───────────────────────────────────
        if self.linea_provisional is not None:
            try:
                self.linea_provisional.remove()
            except ValueError:
                pass
            self.linea_provisional = None

        for a in self.artistas_prov:
            try:
                a.remove()
            except ValueError:
                pass
        self.artistas_prov.clear()

        # ── Datos del nuevo polígono ──────────────────────────────────────────
        vertices = list(self.vertices_actuales)
        clase    = self.clase_actual
        color    = CLASES[clase]
        id_pol   = len(self.poligonos) + 1

        # ── Dibujar patch definitivo ──────────────────────────────────────────
        xy     = np.array(vertices)
        parche = MplPolygon(
            xy, closed=True,
            facecolor=(*color, 0.28),   # Relleno semi-transparente
            edgecolor=color,
            linewidth=2.0,
            zorder=8
        )
        self.ax_img.add_patch(parche)

        # ── Etiqueta centroide ────────────────────────────────────────────────
        cx = float(np.mean(xy[:, 0]))
        cy = float(np.mean(xy[:, 1]))
        texto = self.ax_img.text(
            cx, cy, str(id_pol),
            ha="center", va="center",
            color="white", fontsize=9, fontweight="bold",
            zorder=11,
            bbox=dict(
                facecolor=color, alpha=0.85,
                edgecolor="white", linewidth=0.5,
                boxstyle="round,pad=0.25"
            )
        )

        # ── Análisis espectral (puede tomar unos segundos) ────────────────────
        self._actualizar_status(
            f"⏳ Calculando características del polígono {id_pol}...",
            color="yellow"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        analisis = analizar_poligono(vertices, self.bandas)

        # ── Guardar polígono en el historial ──────────────────────────────────
        self.poligonos.append({
            "id":       id_pol,
            "vertices": vertices,
            "clase":    clase,
            "patch":    parche,
            "texto":    texto,
            "analisis": analisis
        })

        # ── Actualizar tabla de características ───────────────────────────────
        self._agregar_fila_tabla(id_pol, clase, analisis)

        # ── Reiniciar estado de dibujo ────────────────────────────────────────
        self.dibujando         = False
        self.vertices_actuales = []

        # ── Mensaje de estado final ───────────────────────────────────────────
        area_px  = analisis["area_px"]  if analisis else 0
        area_km2 = analisis["area_km2"] if analisis else 0.0

        self._actualizar_status(
            f"✔  Polígono {id_pol} | Clase: {clase} | "
            f"{area_px} px | {area_km2:.4f} km²  —  "
            f"Total polígonos: {len(self.poligonos)}",
            color="lightgreen"
        )
        self.fig.canvas.draw_idle()

    def _agregar_fila_tabla(self, id_pol: int, clase: str, analisis: dict):
        """
        Agrega una nueva fila a la tabla de vectores característicos.

        La fila contiene: ID, clase, área en píxeles, área en km²,
        y la media de reflectancia de cada una de las 7 bandas.
        """
        if analisis is None:
            medias   = [0.0] * 7
            area_px  = 0
            area_km2 = 0.0
        else:
            medias   = analisis["medias"]
            area_px  = analisis["area_px"]
            area_km2 = analisis["area_km2"]

        fila = {
            "ID":       id_pol,
            "Clase":    clase,
            "Area_px":  area_px,
            "Area_km2": round(area_km2, 6)
        }
        for i, nombre in enumerate(NOMBRES_BANDAS):
            fila[nombre] = round(medias[i], 2)

        nueva_fila = pd.DataFrame([fila])
        self.tabla = pd.concat([self.tabla, nueva_fila], ignore_index=True)

    # ────────────────────────────────────────────────────────────────────────
    # CALLBACKS DE BOTONES
    # ────────────────────────────────────────────────────────────────────────

    def _cambiar_clase(self, etiqueta: str):
        """Actualiza la clase seleccionada al hacer clic en un RadioButton."""
        self.clase_actual = etiqueta
        color = CLASES[etiqueta]
        self._actualizar_status(
            f"Clase seleccionada: {etiqueta}  "
            "(presiona Nuevo Polígono si aún no has iniciado uno)",
            color=color
        )

    def _nuevo_poligono(self, event=None):
        """
        Activa el modo de dibujo para un nuevo polígono.
        Si había un polígono en curso, lo descarta.
        """
        # Cancelar dibujo previo si existe
        if self.linea_provisional is not None:
            try:
                self.linea_provisional.remove()
            except ValueError:
                pass
            self.linea_provisional = None

        for a in self.artistas_prov:
            try:
                a.remove()
            except ValueError:
                pass
        self.artistas_prov.clear()

        # Activar modo dibujo
        self.dibujando         = True
        self.vertices_actuales = []

        self._actualizar_status(
            f"🖊  Dibujando polígono — Clase: {self.clase_actual}.  "
            "Clic izquierdo para agregar vértices, doble clic para cerrar.",
            color="lightcyan"
        )
        self.fig.canvas.draw_idle()

    def _limpiar_ultimo(self, event=None):
        """
        Elimina el último polígono completado, o cancela el polígono
        en curso si se está dibujando.
        """
        # Caso 1: hay un polígono en curso → cancelarlo
        if self.dibujando and self.vertices_actuales:
            if self.linea_provisional is not None:
                try:
                    self.linea_provisional.remove()
                except ValueError:
                    pass
                self.linea_provisional = None

            for a in self.artistas_prov:
                try:
                    a.remove()
                except ValueError:
                    pass
            self.artistas_prov.clear()

            self.dibujando         = False
            self.vertices_actuales = []

            self._actualizar_status("Polígono en curso cancelado.",
                                     color="orange")
            self.fig.canvas.draw_idle()
            return

        # Caso 2: no hay polígonos guardados
        if not self.poligonos:
            self._actualizar_status("No hay polígonos para eliminar.",
                                     color="orange")
            return

        # Caso 3: eliminar el último polígono completado
        ultimo = self.poligonos.pop()

        try:
            ultimo["patch"].remove()
            ultimo["texto"].remove()
        except ValueError:
            pass

        # Eliminar la última fila de la tabla
        if len(self.tabla) > 0:
            self.tabla = self.tabla.iloc[:-1].copy()

        self._actualizar_status(
            f"Polígono {ultimo['id']} eliminado.  "
            f"Quedan {len(self.poligonos)} polígono(s).",
            color="orange"
        )
        self.fig.canvas.draw_idle()

    def _limpiar_todo(self, event=None):
        """Elimina todos los polígonos y reinicia la tabla de características."""
        # Cancelar cualquier polígono en curso
        if self.dibujando:
            if self.linea_provisional is not None:
                try:
                    self.linea_provisional.remove()
                except ValueError:
                    pass
                self.linea_provisional = None

            for a in self.artistas_prov:
                try:
                    a.remove()
                except ValueError:
                    pass
            self.artistas_prov.clear()

            self.dibujando         = False
            self.vertices_actuales = []

        # Eliminar todos los patches y textos
        for pol in self.poligonos:
            try:
                pol["patch"].remove()
                pol["texto"].remove()
            except ValueError:
                pass

        self.poligonos = []
        self.tabla     = pd.DataFrame(columns=COLUMNAS_TABLA)

        self._actualizar_status(
            "🗑  Todos los polígonos eliminados. Tabla reiniciada.",
            color="orange"
        )
        self.fig.canvas.draw_idle()

    def _exportar_csv(self, event=None):
        """
        Exporta la matriz de vectores característicos a un archivo CSV.

        El CSV se guarda en dos ubicaciones:
          1. Mismo directorio que la imagen Landsat
          2. Carpeta outputs del taller (/sessions/.../outputs/)

        El CSV puede ser importado en Excel, QGIS, scikit-learn, etc.
        """
        if self.tabla.empty:
            self._actualizar_status(
                "⚠  No hay datos para exportar. "
                "Dibuja al menos un polígono primero.",
                color="orange"
            )
            return

        nombre_csv = "vectores_caracteristicos_ROI.csv"

        # Rutas de destino
        rutas_destino = []

        # 1. Directorio de la imagen
        dir_imagen = os.path.dirname(self.ruta_imagen)
        if dir_imagen and os.path.isdir(dir_imagen):
            rutas_destino.append(os.path.join(dir_imagen, nombre_csv))

        # 2. Carpeta outputs del taller
        dir_outputs = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )
        rutas_destino.append(os.path.join(dir_outputs, nombre_csv))

        # También intentar guardar en la carpeta de outputs de la sesión
        dir_session = "/sessions/determined-lucid-bell/mnt/outputs"
        if os.path.isdir(dir_session):
            rutas_destino.append(os.path.join(dir_session, nombre_csv))

        guardado_en = []
        for ruta in rutas_destino:
            try:
                directorio = os.path.dirname(ruta)
                if directorio:
                    os.makedirs(directorio, exist_ok=True)
                self.tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
                guardado_en.append(ruta)
                print(f"[CSV] Guardado en: {ruta}")
            except Exception as e:
                print(f"[WARN] No se pudo guardar en {ruta}: {e}")

        if guardado_en:
            self._actualizar_status(
                f"✔  CSV exportado — {len(self.tabla)} vectores → "
                f"{os.path.basename(guardado_en[0])}",
                color="lightgreen"
            )
            print(f"\n[EXPORTADO] Tabla con {len(self.tabla)} polígonos:")
            print(self.tabla.to_string(index=False))
        else:
            self._actualizar_status(
                "✗  Error al guardar el CSV. Verifica permisos de escritura.",
                color="red"
            )

    def _mostrar_tabla(self, event=None):
        """
        Muestra la matriz de características en consola
        y abre una ventana secundaria con la tabla visual.
        """
        if self.tabla.empty:
            print("[INFO] La tabla está vacía. Dibuja polígonos primero.")
            self._actualizar_status(
                "Tabla vacía — dibuja al menos un polígono primero.",
                color="orange"
            )
            return

        # Imprimir en consola (útil para depuración)
        print("\n" + "═" * 95)
        print("  MATRIZ DE VECTORES CARACTERÍSTICOS — Polígonos ROI — Landsat 8")
        print("═" * 95)
        print(self.tabla.to_string(index=False))
        print("═" * 95)
        print(f"  Total: {len(self.tabla)} polígono(s)  |  "
              f"Clases presentes: {self.tabla['Clase'].unique().tolist()}\n")

        # Abrir ventana visual con la tabla
        self._mostrar_ventana_tabla()
        self._actualizar_status(
            f"Tabla mostrada — {len(self.tabla)} polígono(s).",
            color="lightgreen"
        )

    def _mostrar_ventana_tabla(self):
        """
        Abre una figura secundaria que muestra la tabla de datos
        con colores por clase para mejor lectura.
        """
        if self.tabla.empty:
            return

        n_filas = len(self.tabla)
        alto    = max(3.0, n_filas * 0.50 + 2.0)

        fig_t, ax_t = plt.subplots(
            figsize=(14, alto),
            facecolor="#1a1a2e",
            num="Vectores Característicos"
        )
        ax_t.set_facecolor("#1a1a2e")
        ax_t.axis("off")

        # Colores de celda según la clase del polígono
        colores_filas = []
        for _, row in self.tabla.iterrows():
            clase  = row["Clase"]
            c      = CLASES.get(clase, (0.3, 0.3, 0.3))
            fila_c = [(*c, 0.22)] * len(COLUMNAS_TABLA)
            colores_filas.append(fila_c)

        # Crear tabla matplotlib
        datos_str = self.tabla.copy()
        for col in NOMBRES_BANDAS + ["Area_km2"]:
            datos_str[col] = datos_str[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
            )

        tabla_mpl = ax_t.table(
            cellText=datos_str.values,
            colLabels=self.tabla.columns.tolist(),
            cellColours=colores_filas,
            colColours=["#0f3460"] * len(COLUMNAS_TABLA),
            loc="center",
            cellLoc="center"
        )
        tabla_mpl.auto_set_font_size(False)
        tabla_mpl.set_fontsize(8.5)
        tabla_mpl.scale(1.0, 1.5)

        # Estilizar encabezados
        for j in range(len(COLUMNAS_TABLA)):
            cell = tabla_mpl[0, j]
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#0f3460")

        ax_t.set_title(
            f"Matriz de Vectores Característicos  —  "
            f"{n_filas} polígono(s) ROI  —  Landsat 8",
            color="white", fontsize=11, pad=18
        )

        fig_t.tight_layout(pad=1.5)
        fig_t.show()

    # ────────────────────────────────────────────────────────────────────────
    # UTILIDADES
    # ────────────────────────────────────────────────────────────────────────

    def _actualizar_status(self, mensaje: str, color: str = "lightgreen"):
        """
        Actualiza el texto de la barra de estado inferior.

        Parámetros
        ----------
        mensaje : str  — texto a mostrar
        color   : str  — color del texto (nombre matplotlib o RGB string)
        """
        self.texto_status.set_text(mensaje)
        self.texto_status.set_color(color)
        self.fig.canvas.draw_idle()

    def mostrar(self):
        """
        Muestra la ventana principal de la aplicación
        y entra al bucle de eventos de matplotlib.
        Este método bloquea hasta que el usuario cierra la ventana.
        """
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Función principal del programa.

    1. Carga la imagen Landsat 8 aplicando el recorte configurado
    2. Construye la interfaz interactiva
    3. Entra al bucle de eventos (espera interacción del usuario)
    """
    print("=" * 62)
    print("  Clasificador ROI — Landsat 8  |  Maestría Proc. Imágenes")
    print("  Universidad Distrital Francisco José de Caldas")
    print("=" * 62)
    print(f"  Imagen: {RUTA_IMAGEN}")
    print(f"  Recorte: filas {FILA_INICIO}–{FILA_FIN}, "
          f"cols {COL_INICIO}–{COL_FIN}")
    print("=" * 62 + "\n")

    # ── Cargar imagen ─────────────────────────────────────────────────────
    try:
        bandas, perfil = cargar_imagen(RUTA_IMAGEN)
    except FileNotFoundError as e:
        print(e)
        print("\nSugerencia: edita la variable RUTA_IMAGEN "
              "al inicio del script con la ruta correcta.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Problema al cargar la imagen: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Lanzar aplicación ─────────────────────────────────────────────────
    app = ClasificadorROI(bandas, perfil, RUTA_IMAGEN)
    app.mostrar()


if __name__ == "__main__":
    main()

# Clasificador ROI — Landsat 8
**Taller: Clasificación Supervisada mediante Dibujo de Regiones de Interés (ROI)**  
Universidad Distrital Francisco José de Caldas — Maestría en Ciencias de la Información y las Comunicaciones  
Métodos Avanzados en Análisis de Imágenes  
Sebastián Morales Tarapues — Cód. 20182020039

---

## ¿Qué hace?

Herramienta interactiva en Python para definir muestras de entrenamiento sobre imágenes satelitales Landsat 8. El usuario dibuja polígonos directamente sobre la imagen y la aplicación calcula automáticamente el **vector de características** de cada ROI (media de DN por banda espectral), generando la matriz de entrenamiento lista para usar en algoritmos de clasificación supervisada.

---

## Cómo descargar y ejecutar el proyecto paso a paso

### Paso 1 — Descargar el código

Tienes dos opciones:

**Opción A (más fácil, sin necesidad de Git):**
1. Haz clic en el botón verde **`< > Code`** en la parte superior de esta página
2. Selecciona **`Download ZIP`**
3. Descomprime el archivo ZIP en una carpeta de tu preferencia (por ejemplo `C:\ROI\`)

**Opción B (con Git):**
```bash
git clone https://github.com/SebastianMT512/ROI.git
```

---

### Paso 2 — Verificar que tienes Python instalado

Abre una terminal (en Windows: busca **cmd** o **PowerShell** en el menú inicio) y escribe:

```bash
python --version
```

Deberías ver algo como `Python 3.10.x`. Si no tienes Python, descárgalo desde https://www.python.org/downloads/ (versión 3.8 o superior).

---

### Paso 3 — Instalar las bibliotecas necesarias

En la misma terminal, ejecuta este comando para instalar todo de una vez:

```bash
pip install rasterio numpy pandas matplotlib shapely
```

Espera a que termine la instalación (puede tardar unos minutos).

---

### Paso 4 — Colocar la imagen Landsat

Copia el archivo de imagen **`LC08_L1TP_009058_20251208_20251217_02_T1_Bandas1-7.tif`** dentro de la misma carpeta donde descomprimiste el proyecto, al lado del archivo `clasificador_roi.py`.

> ⚠️ El archivo `.tif` no está incluido en el repositorio por su tamaño.  
> Puedes descargarlo desde el siguiente enlace:  
> [Descargar imagen Landsat 8](https://udistritaleduco-my.sharepoint.com/:i:/g/personal/smoralest_udistrital_edu_co/IQAe4vWjmB9RTqriUzIKiNHbAQtUVusKoMEvNQnGxsJa0GA?e=QZoTuP)  
> Una vez descargado, colócalo en la misma carpeta que `clasificador_roi.py`.

---

### Paso 5 — Ejecutar el programa

En la terminal, navega hasta la carpeta del proyecto:

```bash
cd C:\ROI
```

Y ejecuta el script:

```bash
python clasificador_roi.py
```

Se abrirá automáticamente una ventana con la imagen Landsat 8 lista para usar.

---

## Cómo usar la herramienta

Una vez abierta la ventana:

1. Selecciona una clase en el panel izquierdo (Agua, Cultivos, Bosque, etc.)
2. Haz clic en el botón **[Nuevo Polígono]**
3. Haz clic sobre la imagen para agregar los vértices del polígono
4. Haz **doble clic** para cerrar y finalizar el polígono
5. Repite los pasos 1–4 para cada zona de interés
6. Presiona **[Mostrar Tabla]** para ver la matriz de vectores característicos
7. Presiona **[Exportar CSV]** para guardar los resultados

| Botón | Función |
|-------|---------|
| **[Nuevo Polígono]** | Inicia un nuevo ROI |
| **[Limpiar Último]** | Elimina el último polígono dibujado |
| **[Limpiar Todo]** | Elimina todos los polígonos |
| **[Mostrar Tabla]** | Muestra la matriz de vectores característicos |
| **[Exportar CSV]** | Guarda los resultados en un archivo CSV |

---

## Archivo de salida

El botón **[Exportar CSV]** genera `vectores_caracteristicos_ROI.csv` en la carpeta de la imagen con la siguiente estructura (una fila por polígono, una columna por banda):

| ID | Clase | Area_px | Area_km2 | B1_mean | B2_mean | B3_mean | B4_mean | B5_mean | B6_mean | B7_mean |
|----|-------|---------|----------|---------|---------|---------|---------|---------|---------|---------|
| 0 | Agua | 312 | 0.28 | ... | ... | ... | ... | ... | ... | ... |

---

## Estructura del repositorio

```
ROI/
├── clasificador_roi.py     # Script principal
├── README.md               # Este archivo
└── LC08_...tif             # Imagen Landsat 8 (no incluida — ver Paso 4)
```

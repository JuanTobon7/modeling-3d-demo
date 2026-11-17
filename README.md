# modeling-3d-demo · Scanner3D

Escáner 3D por fotogrametría: tomas 20–60 fotos alrededor de un objeto (a mano o en un plato
giratorio) y el servidor en Python reconstruye un modelo 3D con color (`.glb`, `.obj`, `.ply`).
El cliente para celular es una PWA que sirve el mismo servidor: no hay que compilar nada ni
pasar por una tienda de apps.

```
Celular (PWA en el navegador)  ── HTTP/HTTPS ──►  Servidor FastAPI (Python)
 • cámara guiada (brújula / temporizador)          • API REST + cola de trabajos
 • o fotos de la galería                           • cada reconstrucción en su propio proceso
 • visor 3D (three.js)        ◄── GLB/OBJ/PLY ──   • pipeline de fotogrametría
```

## Inicio rápido

Requiere Python 3.10–3.12 (Open3D y pycolmap publican ruedas para esas versiones).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python run.py --https
```

La terminal muestra una URL y un código QR. Ábrelo con el celular conectado **a la misma red
Wi-Fi**. La primera vez, acepta la advertencia del certificado (es autofirmado).

- `python run.py` (sin `--https`) también funciona, pero los navegadores móviles solo permiten la
  cámara en vivo con HTTPS; en ese modo se usan fotos de la galería.
- El enlace lleva un **token** aleatorio; sin él la API responde 401. Opciones: `--token MI_TOKEN`,
  `--no-token` (solo en redes de confianza), `--port 8000`, `--data carpeta`.
- En Windows, si la ruta del proyecto es muy larga, Open3D puede fallar al cargar sus DLL
  ("nombre de archivo demasiado largo"): crea el entorno virtual en una ruta corta, por ejemplo `C:\venvs\scanner3d`.

Sin servidor, desde la terminal:

```bash
python -m scanner3d.cli carpeta_con_fotos salida/ --engine auto --quality normal
```

## Cómo tomar las fotos

- Rodea el objeto completo, una foto cada ~10° (36 fotos por vuelta). Si puedes, da una segunda
  vuelta más alta o más baja.
- Objeto **con textura** (no liso, ni brillante, ni transparente), buena luz difusa, sin moverlo.
- Que el objeto ocupe buena parte de la foto y que haya solape entre fotos consecutivas.

Modos de captura en la app:

| Modo | Uso |
|---|---|
| Manual | Tocas el disparador en cada posición. |
| Brújula | Dispara solo cuando entras en un sector de 10° que aún no tiene foto. El anillo muestra la cobertura. |
| Temporizador | Celular fijo + plato giratorio: dispara cada 1–5 s. |

## Pipeline

1. **Preproceso** (`pipeline/preprocess.py`): orientación EXIF, redimensionado y focal a partir del EXIF
   (equivalente 35 mm); si no hay EXIF, se estima y luego se refina.
2. **Características** (`pipeline/features.py`): SIFT (pirámide de convoluciones gaussianas) con
   RootSIFT, emparejamiento con test de Lowe y verificación geométrica con la matriz esencial (USAC/RANSAC).
3. **Structure-from-Motion**: dos motores intercambiables.
   - `colmap` (`pipeline/sfm_colmap.py`): pycolmap, el más robusto. Es el predeterminado.
   - `opencv` (`pipeline/sfm_opencv.py`): SfM incremental propio: par inicial, PnP-RANSAC,
     triangulación y *bundle adjustment* con SciPy (incluye la focal si no viene en el EXIF).
   - En modo `auto`, si COLMAP falla o no está instalado, pasa solo al motor OpenCV.
4. **Densificación** (`pipeline/dense.py`): rectificación estéreo entre vistas vecinas, SGBM,
   chequeo izquierda-derecha y consistencia geométrica entre varias vistas; recorte a la región del objeto.
5. **Malla** (`pipeline/mesh.py`): Open3D: limpieza de outliers, normales orientadas hacia las cámaras,
   Poisson, recorte por densidad, eliminación de fragmentos sueltos y color por vértice.
6. **Exportación** (`pipeline/export.py`): alinea el modelo (arriba = +Y, apoyado en y=0, tamaño 1) y
   guarda `model.glb`, `model.obj`, `model.ply`, `points.ply` y `cameras.json`.

Calidades: `fast` (1200 px), `normal` (1600 px) y `high` (2400 px, Poisson más fino).

## API

Todas las rutas `/api/*` (excepto `/api/health`) requieren el token en la cabecera `X-Token`
o en el parámetro `?token=`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/info` | Motores disponibles y límites |
| GET / POST | `/api/jobs` | Listar / crear (`{"name", "engine": auto\|colmap\|opencv, "quality": fast\|normal\|high}`) |
| GET | `/api/jobs/{id}` | Estado, progreso, resultado y archivos |
| POST | `/api/jobs/{id}/images` | Subir fotos (multipart, campo `files`, se pueden mandar por lotes) |
| POST | `/api/jobs/{id}/start` · `/cancel` | Iniciar (o reintentar) / cancelar |
| DELETE | `/api/jobs/{id}` | Borrar trabajo y fotos |
| GET | `/api/jobs/{id}/files/{nombre}` | Descargar `model.glb`, `model.obj`, `model.ply`, `points.ply` |

Las fotos se validan (formato real JPEG/PNG/WEBP/BMP/TIFF, máximo 30 MB y 300 fotos; configurable
con `SCANNER_MAX_MB` y `SCANNER_MAX_IMAGES`).

## Pruebas

```bash
pytest -q
```

Generan un dataset sintético (un objeto con textura renderizado desde N ángulos, `scanner3d/synthetic.py`)
y comprueban: la API (token, validación de fotos, cancelación, borrado, un trabajo de punta a punta),
que cada motor recupere las cámaras con error < 2 % tras alinear con la verdad, y que el pipeline
completo produzca una malla normalizada. Para generar el dataset a mano:

```bash
python -m scanner3d.synthetic fotos_prueba/ --views 36
```

## Licencias

Todas las dependencias son permisivas (ninguna GPL/AGPL):

| Componente | Licencia |
|---|---|
| FastAPI, Open3D, trimesh, three.js | MIT |
| OpenCV, python-multipart, cryptography | Apache-2.0 |
| Uvicorn, NumPy, SciPy, pycolmap/COLMAP, qrcode | BSD |
| Pillow | HPND (tipo MIT) |

Notas: las ruedas precompiladas de OpenCV incluyen FFmpeg (LGPL, enlazado dinámicamente) y las de
pycolmap incluyen bibliotecas de runtime de GCC; ninguna obliga a liberar tu código. Si quieres evitar
incluso eso, no instales pycolmap y usa el motor `opencv`. Se descartaron OpenMVS (AGPL) y Meshroom (MPL-2.0).

## Limitaciones

- Probado con datos sintéticos y con la interfaz en un navegador con cámara simulada; **aún no con
  fotos reales de un celular**.
- La brújula del navegador no está verificada en un teléfono real; si no hay brújula, quedan el disparo
  manual y el temporizador.
- El motor OpenCV es más lento y menos limpio que COLMAP (puede dejar huecos en zonas vistas desde pocas fotos).
- El visor carga three.js desde un CDN (jsDelivr): el celular necesita internet para ver el modelo
  (la reconstrucción y las descargas no lo necesitan).

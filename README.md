# TC5061 Proyecto Equipo 19: MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Propósito

Propuesta y base reproducible de Machine Learning para el sistema de bicicleta pública
[MiBici](https://www.mibici.net/) de Guadalajara. El repositorio contiene la ingesta
versionada de millones de viajes, el módulo de feature engineering y un experimento base
registrado en MLflow Tracking que valida la viabilidad técnica de la propuesta.

## Preguntas

| | Pregunta | Tarea |
|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering |

El ML Canvas y el diagrama de arquitectura están en [`references/`](references/).

## Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Los enlaces
  exactos de cada archivo están en `tc5061_proyecto_equipo19/config.py` (`URLS_VIAJES`,
  `URL_ESTACIONES`).
- **Rango:** enero 2024 a diciembre 2025 (24 meses completos, dos ciclos anuales para
  capturar la estacionalidad).
- **Tamaño obtenido:** 9,177,953 viajes y 484 estaciones en la nomenclatura `nomenclatura_2026_09.csv`.
- **Variables usadas:** `trip_id`, `start_time`, `end_time`, `origin_station_id`,
  `destination_station_id` y, del catálogo, `station_id`, `station_name`, `latitude`,
  `longitude`, `capacity`. Detalle en [`references/data_dictionary.md`](references/data_dictionary.md).
- **Datos personales:** `Usuario_Id`, `Genero` y `Año_de_nacimiento` se descartan en la ingesta y
  nunca llegan a los Parquet. Los datos crudos y procesados no se suben a Git.
- **Partición temporal** (definida en `config.py`): entrenamiento 2024-01 a 2025-06, validación
  2025-07 a 2025-09 y prueba 2025-10 a 2025-12. Las variables solo usan información anterior al
  momento de la predicción y las estadísticas se calculan solo con el periodo de entrenamiento.
- **Versionado:** `references/data_manifest.json` guarda, por archivo, el número de registros y el
  hash SHA-256 del CSV original y del Parquet. Su campo `data_version` es un hash corto de todos
  los Parquet; cada run de MLflow lo registra como parámetro y etiqueta, y adjunta el manifiesto
  como artefacto. Si cambian los datos, cambia `data_version` y el cambio queda en el historial de Git.

## Estructura de carpetas

```
├── Makefile           <- Comandos: make data, make features, make features_p2, make train_p2, make mlflow-ui
├── README.md
├── data               <- No se versiona en Git
│   ├── raw            <- CSV originales descargados de MiBici
│   ├── interim        <- Viajes limpios y enriquecidos
│   └── processed      <- Parquet normalizados (trips_AAAA_MM.parquet, stations.parquet)
├── docs               <- Proyecto de mkdocs
├── models             <- Modelos serializados (no se versionan)
├── notebooks          <- Notebooks de exploración (número-iniciales-descripción)
├── pyproject.toml     <- Metadatos del paquete y configuración de ruff
├── references         <- ML Canvas, diagrama de arquitectura, diccionario y manifiesto de datos
├── reports
│   └── figures        <- Figuras generadas por el experimento
├── requirements.txt   <- Dependencias con versiones fijadas
└── tc5061_proyecto_equipo19
    ├── config.py      <- Rutas, fechas, particiones, semillas, URLs y configuración de MLflow
    ├── dataset.py     <- Ingesta: descarga, normalización, Parquet y manifiesto
    ├── features/      <- Paquete de feature engineering reutilizable
        ├── base.py    <- Carga, limpieza, calendario y partición temporal (compartido)
        ├── p1.py      <- Demanda diaria por estación (P1)
        ├── p2.py      <- Variables y etiqueta de riesgo (P2)
        └── p3.py      <- Perfil de uso por estación (P3)
    ├── plots.py
    └── modeling
        ├── mlflow_utils.py <- Parámetros, etiquetas y artefactos comunes de cada run
        ├── train_p2.py     <- Baselines y candidatos de P2 registrados en MLflow
        ├── train.py
        └── predict.py
```

## Instalación

Requiere Python 3.14.

```bash
git clone https://github.com/A01797947/TC5061-Proyecto-Equipo19.git
cd TC5061-Proyecto-Equipo19
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Ingesta de datos

Descarga los 24 meses y la nomenclatura, normaliza columnas, codificación y fechas, guarda Parquet
y regenera el manifiesto. Los archivos ya descargados no se vuelven a bajar.

```bash
make data
# equivalente a:
python -m tc5061_proyecto_equipo19.dataset
# por etapas:
python -m tc5061_proyecto_equipo19.dataset descargar
python -m tc5061_proyecto_equipo19.dataset procesar
```

Al terminar, compara `references/data_manifest.json` con el de `main`: si `data_version` coincide,
tienes exactamente los mismos datos que el resto del equipo.

## Experimento en MLflow

Todos los runs van al experimento `mibici-equipo19`, guardado en `mlflow.db` (SQLite) con
artefactos en `mlartifacts/`. Para usar un servidor de MLflow compartido, define
`MLFLOW_TRACKING_URI` en un archivo `.env`.

`tc5061_proyecto_equipo19/modeling/mlflow_utils.py` da a cada run lo que necesita para ser
reproducible:

- **Parámetros:** rango de fechas, cortes de entrenamiento, validación y prueba, `data_version`,
  versión del esquema, número de meses y viajes y semilla, más los del modelo.
- **Etiquetas:** `equipo`, `pregunta`, `tipo_modelo` (baseline o candidato), `git_commit`,
  `git_branch`, `git_dirty` y `data_version`.
- **Artefactos:** el manifiesto de datos, al que cada run agrega sus figuras.

Ejecuta los runs con el árbol de trabajo limpio (sin cambios sin commit) para que `git_commit`
identifique exactamente el código usado; `git_dirty=True` indica lo contrario.

### Capturas de evidencia

```bash
make mlflow-ui
# equivalente a:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abre http://127.0.0.1:5000 y toma estas capturas:

1. **Lista de runs:** experimento `mibici-equipo19` en la pestaña *Runs*, con las columnas de
   métricas y la etiqueta `pregunta` visibles.
2. **Detalle de un run:** las secciones *Parameters*, *Metrics* y *Tags* de un run.
3. **Artefacto:** la pestaña *Artifacts* de un run con una figura abierta.
4. **Comparación:** selecciona dos o más runs y pulsa *Compare*; captura la tabla de parámetros y
   métricas.

## P2. Riesgo de estación sin bicicletas o sin anclajes

Clasificación por estación y hora: ¿la estación quedará vacía o llena en las próximas tres horas? La aproximación del riesgo, el pipeline, los modelos, las métricas y lo que se registra en MLflow están en [`references/p2_riesgo_estaciones.md`](references/p2_riesgo_estaciones.md).

```bash
make features_p2   # python -m tc5061_proyecto_equipo19.features.p2: tabla de variables y etiqueta
make train_p2      # python -m tc5061_proyecto_equipo19.modeling.train_p2: baselines y candidatos en MLflow
```

## Equipo

| Integrante | Rol |
|---|---|
| José Llashag Huaman | Por definir |
| Ismael Ortiz Díaz | Por definir |
| José Martín Ramos Requena | Por definir |
| Denisse Martínez Ruíz | Por definir |

## Flujo de trabajo con Git y GitHub

Seguimos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): un flujo ligero donde `main` siempre está estable y todo cambio nace en una rama corta y entra por Pull Request.

### Principios

1. **`main` es la única rama permanente** y siempre contiene código que funciona.
2. **Nadie hace push directo a `main`.** Todo cambio entra mediante Pull Request.
3. **Trabajamos en ramas en el mismo repositorio, sin forks.** Los integrantes son colaboradores con permiso de escritura.
4. **Toda rama sale de `main` actualizada** y vive poco tiempo.
5. **Cada Pull Request lo revisa otra persona** antes de integrarse.
6. **Los Pull Requests son pequeños y enfocados** en un solo cambio.

### Protección de la rama `main`

La rama `main` está protegida (Settings → Branches → Branch protection rules) con estas reglas:

| Regla | Efecto |
|---|---|
| Require a pull request before merging | No se puede hacer push directo |
| Required approvals: 1 | Alguien distinto al autor debe aprobar |
| Dismiss stale pull request approvals | Un commit nuevo invalida aprobaciones previas |
| Require conversation resolution | Los comentarios deben resolverse antes del merge |
| Block force pushes y deletions | El historial de `main` no se reescribe ni se borra |
| Do not allow bypassing the above settings | Las reglas también aplican al dueño del repo y a los administradores: nadie puede hacer push directo a `main` ni hacer merge sin la aprobación de otra persona |

### Convención de nombres de ramas

Formato: `tipo/descripcion-corta`, en minúsculas y separando palabras con guiones.

| Tipo | Uso | Ejemplo |
|---|---|---|
| `feat` | Funcionalidad nueva | `feat/ingesta-parquet` |
| `fix` | Corrección de un error | `fix/encoding-2024` |
| `exp` | Modelos y experimentos | `exp/p1-baseline-regresion` |
| `docs` | Documentación | `docs/readme-flujo-git` |
| `chore` | Mantenimiento del repositorio | `chore/ampliar-gitignore` |

### Convención de mensajes de commit

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
tipo(alcance): descripción en imperativo y minúsculas
```

- **Tipos:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- **Alcances:** `dataset`, `features`, `p1`, `p2`, `p3`, `mlflow`, `repo`, `docs`.
- La descripción va en imperativo, sin punto final y de preferencia con menos de 72 caracteres.

Ejemplos:

```
feat(dataset): normalizar nombres de columnas entre años
fix(features): evitar fuga de información en lags
feat(p3): agregar mapa de estaciones por cluster
chore(repo): ampliar gitignore con mlruns y mlartifacts
docs(readme): documentar comandos de ingesta
```

### Paso a paso de un cambio

**Paso 0. Configuración inicial (una sola vez)**

```bash
git clone https://github.com/A01797947/TC5061-Proyecto-Equipo19.git
cd TC5061-Proyecto-Equipo19
git config user.name "Tu Nombre"
git config user.email "tu-correo@ejemplo.com"
```

**Paso 1. Actualizar `main` local**

```bash
git checkout main
git pull origin main
```

**Paso 2. Crear la rama de trabajo**

```bash
git checkout -b feat/ingesta-parquet main
```

**Paso 3. Trabajar y hacer commits pequeños y frecuentes**

```bash
git status
git add tc5061_proyecto_equipo19/dataset.py
git commit -m "feat(dataset): descargar meses y nomenclatura de estaciones"
```

Agrega archivos concretos con `git add <archivo>` y evita `git add .`, para no subir datos, `.env`, `mlruns/` ni modelos por accidente.

**Paso 4. Subir la rama**

```bash
git push -u origin feat/ingesta-parquet
```

Las ramas de trabajo no están protegidas, así que cada quien puede hacer push libremente a la suya.

**Paso 5. Mantener la rama al día**

Si `main` avanzó mientras trabajabas:

```bash
git fetch origin
git merge origin/main
```

Si hay conflictos, resuélvelos en tu rama (ver *Resolver conflictos*), haz commit y vuelve a hacer push.

**Paso 6. Abrir el Pull Request**

En GitHub: Compare & pull request, con base `main` y compare tu rama. Usa:

- **Título:** el mismo formato de Conventional Commits, por ejemplo `feat(dataset): ingesta reproducible a Parquet`.
- **Descripción:** qué cambia, por qué y cómo probarlo.
- **Reviewer:** el siguiente en la rotación (ver *Rotación de revisores*).
- Marca el PR como Draft si aún no está listo.

**Paso 7. Revisión**

El revisor lee los cambios y comenta o aprueba (ver *Rotación de revisores* y *Qué revisa el revisor*). El autor responde y corrige con nuevos commits en la misma rama. Al hacer push, el PR se actualiza solo.

**Paso 8. Merge**

Con la aprobación y las conversaciones resueltas, se usa **Squash and merge**, que deja un commit limpio en `main` con el título del PR. Después se elimina la rama remota (botón Delete branch).

**Paso 9. Limpiar localmente**

```bash
git checkout main
git pull origin main
git branch -d feat/ingesta-parquet
```

### Rotación de revisores

Cada integrante hace al menos un Pull Request propio y al menos una revisión a otro integrante. Para repartir las revisiones usamos una rotación circular:

```
José Llashag Huaman → Ismael Ortiz Díaz → José Martín Ramos Requena → Denisse Martínez Ruíz → José Llashag Huaman
```

Quien abre un PR lo asigna a la siguiente persona en la rotación. Si el revisor asignado no está disponible, cualquier otro integrante puede revisarlo.


### Qué revisa el revisor

- [ ] El código corre y hace lo que dice el PR.
- [ ] No hay datos de MiBici, credenciales, `.env`, `mlruns/` ni modelos serializados en el commit.
- [ ] Las rutas, fechas y semillas vienen de `config.py` y no están escritas a mano.
- [ ] Las variables no usan información posterior al momento de la predicción.
- [ ] Las funciones nuevas tienen docstring que explica qué hacen y por qué son pertinentes.
- [ ] Los nombres de la rama y los commits siguen la convención.
- [ ] El PR cubre una sola cosa y es fácil de leer.

Cada revisión termina en una de tres acciones: Approve, Comment o Request changes.

### Resolver conflictos

Un conflicto aparece cuando dos ramas modifican las mismas líneas. Se resuelve en la rama propia, nunca en `main`:

```bash
git fetch origin
git merge origin/main
# Git marca los archivos en conflicto con <<<<<<<, ======= y >>>>>>>
# Edita cada archivo, deja la versión correcta y quita los marcadores
git add <archivo-resuelto>
git commit
git push
```

VS Code (con la extensión GitLens) muestra los conflictos y permite elegir entre los cambios entrantes, los actuales o ambos. Si hay duda sobre cuál versión conservar, hablen con el autor del otro cambio antes de decidir.

### Buenas prácticas

- Haz commits pequeños con un propósito claro.
- Abre el PR temprano, como Draft, si quieres retroalimentación a medio camino.
- Responde las revisiones a tiempo para no bloquear al equipo.
- No reescribas el historial de una rama que ya se compartió.
- Si algo salió mal en `main`, se corrige con un nuevo PR (por ejemplo `fix/...` o `git revert`), nunca con force push.
- Los modelos y las figuras generadas se suben solo cuando el equipo lo decide, y nunca los datos crudos.

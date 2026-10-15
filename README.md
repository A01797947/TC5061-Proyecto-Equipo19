# TC5061 Proyecto Equipo 19

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto MiBici: propuesta y base reproducible

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


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tc5061_proyecto_equipo19 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tc5061_proyecto_equipo19   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tc5061_proyecto_equipo19 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


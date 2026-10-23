# Diccionario de datos de MiBici

## 1. Datos de viajes

**Fuente:** portal oficial de datos abiertos de MiBici.

**Archivo de origen:** `datos_abiertos_AAAA_MM.csv`

| Campo canónico | Descripción | Tipo |
|---|---|---|
| `trip_id` | Identificador del viaje | Cadena |
| `start_time` | Fecha y hora de inicio | Fecha y hora |
| `end_time` | Fecha y hora de fin | Fecha y hora |
| `origin_station_id` | Identificador de estación de origen | Cadena |
| `destination_station_id` | Identificador de estación de destino | Cadena |

Los archivos originales contienen campos adicionales relacionados con el usuario. La primera versión analítica excluye dichos campos por no ser necesarios para los objetivos iniciales.

## 2. Datos de estaciones

**Archivo de origen:** `nomenclatura_AAAA_MM.csv`

| Campo canónico | Descripción | Tipo |
|---|---|---|
| `station_id` | Identificador de estación | Cadena |
| `station_name` | Nombre de la estación | Texto |
| `latitude` | Latitud | Numérico |
| `longitude` | Longitud | Numérico |
| `capacity` | Capacidad declarada de la estación | Numérico entero cuando corresponda |

## 3. Normalización

Los nombres originales se transforman mediante un mapeo explícito. La variante `AÃ±o_de_nacimiento` se reconoce como alias de `Año_de_nacimiento`, aunque esta variable se excluye del conjunto analítico inicial.

Las fechas inválidas, los identificadores duplicados, las coordenadas faltantes y las inconsistencias de capacidad deben cuantificarse y analizarse antes de decidir si corresponde eliminar o corregir registros.

## 4. Formato procesado

Los viajes se almacenan en archivos Parquet separados por mes. El catálogo de estaciones se almacena por separado. Los archivos originales se preservan sin modificaciones.

## 5. Versionado

El manifiesto registra los hashes SHA-256, el número de registros, el rango temporal disponible y la versión del esquema. Los datos originales y procesados no se almacenan en Git.

El campo `data_version` del manifiesto es un hash corto (12 caracteres) de la versión del esquema y de los hashes de todos los Parquet. Cada run de MLflow lo registra como parámetro y etiqueta, de modo que un resultado siempre se puede ligar a los datos exactos que lo produjeron.

## 6. Limitaciones

El catálogo actual de estaciones puede no representar todas las estaciones que existieron en el histórico. La capacidad de una estación no representa el inventario real de bicicletas disponibles en un momento determinado.
# data/

Git ignores every file in this folder except this README. Do not commit telematics traces:
they are location data about real people.

## 1. Telematics stream (main input)

| Item | Value |
|---|---|
| Format | Long CSV, one row for each reading |
| Required columns | `deviceId`, `timestamp`, `variable`, `value` (other columns are ignored) |
| Path | `--data <file>` or `DRIVERISK_DATA` |
| Source | Your own fleet export, or a public multi-driver data set. Check its licence and the consent of the drivers |

Signals that the ingest reads (`variable` names are matched without case, spaces or `_`):

| `variable` | Column after ingest | Unit |
|---|---|---|
| `Vehicle speed`, `SPEED` | `speed_kmh` | km/h |
| `ENGINE RPM`, `RPM` | `rpm` | 1/min |
| `ACCELERATION X` | `acc_x` (longitudinal) | m/s² (or g with `DRIVERISK_ACC_UNIT=g`) |
| `ACCELERATION Y` | `acc_y` (lateral) | m/s² (or g) |
| `ACCELERATION Z` | `acc_z` | m/s² (or g) |
| `POSITION` | `lat`, `lon`, `alt` | text `lat,lon` or `lat,lon,alt` in degrees and metres |
| `IGNITION_STATUS` | `ignition` | `1`/`0`, `ON`/`OFF`, `true`/`false` |

Other variables are counted in the ingest report and ignored.

## 2. Weather (optional)

| Source | Setting | Notes |
|---|---|---|
| Synthetic | `DRIVERISK_WEATHER=offline` (default) | For the demo and the tests only |
| Station CSV | `DRIVERISK_WEATHER=csv`, `DRIVERISK_WEATHER_CSV=<file>` | Columns `station_id, lat, lon, time, temp_c, precip_mm, wind_ms`. A station must be within 50 km of the reading |
| Open-Meteo archive API | `DRIVERISK_WEATHER=open-meteo` | <https://open-meteo.com/en/docs/historical-weather-api>, no key. Read the terms of use. One request for each 0.1° cell and day range, cached in `DRIVERISK_CACHE_DIR` |

## 3. Road classes (optional)

`DRIVERISK_ROADS=csv` with `DRIVERISK_ROADS_CSV=<file>`: points along the road network with the columns
`lat, lon, road_class, speed_limit_kmh`. Prepare the file one time from an OpenStreetMap extract
(<https://download.geofabrik.de/>, ODbL licence), for example with `osmium` or `pyrosm`. Use the classes
`motorway`, `primary`, `secondary` and `residential`. A reading farther than 50 m from a point gets `unknown`.

## 4. Collision prior (optional)

`DRIVERISK_COLLISIONS_CSV=<file>`: a collision table with `latitude` and `longitude` columns, for example the
UK Department for Transport road safety data (collision table), <https://www.data.gov.uk/dataset/road-accidents-safety-data>,
Open Government Licence. The *casualty* table has no location, so the code refuses it. The prior is used only
when at least 50 % of the readings are inside the area of the table.

## 5. Synthetic data (no download)

`driverisk simulate --drivers 40 --trips 12 --out data/synthetic_telematics.csv` writes a fake fleet in the
long format above. With no `--data` and no `DRIVERISK_DATA`, each command simulates the fleet in memory.

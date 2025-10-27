# Data: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## 1. Overview

The experiments in this project use publicly available meteorological datasets derived from **ERA5** and the **GenCast Mini** demonstration environment released by DeepMind. The data represent global reanalysis fields used for training and evaluating diffusion-based weather forecasting models.  
The datasets are automatically downloaded when running the **GenCast Mini notebook**, ensuring reproducibility without the need for manual file transfers or authentication.

---

## 2. Data Sources

### 2.1 ERA5 Reanalysis (ECMWF)
- **Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF)  
- **Access:** Copernicus Climate Data Store (CDS)  
- **Description:** Global hourly atmospheric reanalysis dataset, providing consistent observations and model analyses since 1979.  
- **Variables used:** Temperature, geopotential, winds (u/v components), vertical velocity, humidity, mean sea level pressure, and surface parameters (2m temperature, 10m winds, sea surface temperature, total precipitation).

### 2.2 ERA5 Ensemble Data Assimilation (EDA)
- **Purpose:** Provides flow-dependent ensemble members that estimate uncertainty in ERA5 analyses.  
- **Usage in GenCast:** Used indirectly to initialize the ensemble mean fields; GenCast adds Gaussian-process noise to approximate the EDA spread.

### 2.3 GenCast Mini Demonstration Data
- **Provider:** DeepMind (via GenCast GitHub repository)  
- **Access:** Automatically downloaded and cached within the notebook execution environment.  
- **Contents:** Sample input and target datasets for 12-hour forecasting cycles, preprocessed from ERA5.  
- **Format:** NetCDF/xarray datasets with coordinates:


Dimensions:
batch = 1
time = 2 (inputs) + 1 (target)
lat = 181
lon = 360
level = 13 (pressure levels)


---

## 3. Data Description

Each dataset contains multivariate atmospheric fields structured as:
| Variable | Type | Levels | Units |
|-----------|------|---------|--------|
| 2m Temperature | Surface | 1 | K |
| Mean Sea Level Pressure | Surface | 1 | Pa |
| 10m Wind Components (U, V) | Surface | 1 | m/s |
| Sea Surface Temperature | Surface | 1 | K |
| Total Precipitation (12hr) | Surface | 1 | m |
| Temperature | Atmospheric | 13 | K |
| Geopotential | Atmospheric | 13 | m²/s² |
| Wind Components (U, V) | Atmospheric | 13 | m/s |
| Vertical Velocity | Atmospheric | 13 | Pa/s |
| Specific Humidity | Atmospheric | 13 | kg/kg |

These variables collectively describe both the surface and upper-atmospheric conditions required for medium-range forecasts.

---

## 4. Data Access in Notebook

The **GenCast Mini** notebook handles dataset downloads automatically using the following workflow:

1. When executed for the first time, the notebook fetches all required NetCDF files from the official DeepMind storage location.  
2. Data are cached locally in the working directory (e.g., `/tmp/datasets/`) for reuse in subsequent runs.  
3. Users can modify dataset paths or cache directories by changing the configuration in the notebook’s setup cell.  

This automated mechanism ensures that all users can reproduce the same experimental conditions without manual downloads.

---

## 5. Data Preprocessing

- Data are loaded as **xarray Datasets** for ease of manipulation and broadcasting.  
- Variables are normalized to zero mean and unit variance before model inference.  
- Perturbations are applied directly on the normalized fields.  
- Missing or NaN values (e.g., over land for SST) are automatically masked.  
- Latitude and longitude coordinates are preserved for visualization and metric computation.

---

## 6. Licensing and Attribution

- **ERA5 Data:** Licensed under the Copernicus Climate Data Store (CDS) Terms of Use.  
- **GenCast Mini Demo:** Released under the Apache License 2.0 by DeepMind Technologies Limited (2024).  
- **Citation:**  
L. Keet et al., “GenCast: Diffusion-based ensemble forecasting for medium-range weather,” *DeepMind Technical Report*, 2024.

---

## 7. Summary

The data used in this research are entirely open and reproducible.  
The GenCast Mini notebook provides automated access to preprocessed ERA5-based datasets, ensuring consistent evaluation of perturbation experiments across environments.  
This structure eliminates data management overhead and guarantees alignment with the original GenCast evaluation framework.

---

**Note:** No raw data are stored in the repository. Datasets are downloaded dynamically when executing the notebook.



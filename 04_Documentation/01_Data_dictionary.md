# Input data dictionary

## 01_Fangshan_samples.csv

One row per observation. There are 4,403 retained landslide records and 4,403 fixed sampled pseudo-absence locations. Do not interpret the negative class as field-confirmed stable terrain.

| Field | Meaning |
|---|---|
| `sample_id` | Unique observation identifier used to join data and folds |
| `label` | 1 = landslide; 0 = sampled pseudo-absence |
| `x`, `y` | Projected coordinates in metres, EPSG:32650 |
| `Aspect` | Supplied aspect attribute; the modelling code uses the supplied sine/cosine columns |
| `Aspect_sin`, `Aspect_cos` | Cyclic aspect encoding already extracted in the original input |
| `Curvature`, `Elevation`, `TWI`, `Slope` | Terrain attributes supplied with each point |
| `Landcover`, `Soil`, `Lithology` | Categorical codes; encoded by training-fitted one-hot encoders |
| `Distance_to_river`, `Distance_to_road` | Distance attributes used in the 16-factor analysis |
| `FVC`, `NDVI_recent`, `NDVI_mean`, `NDVI_std`, `NDVI_trend` | Supplied vegetation attributes |
| `precipitation_mean`, `precipitation_max` | Supplied precipitation attributes |
| `Distance_to_fault`, `Distance_to_coal_mine`, `Distance_to_seismic_points` | Additional attributes used with lithology in the 20-factor analysis |
| `inventory_cluster` | Fixed positive-inventory cluster identifier; not a fitted validation label |
| `spatial_group` | Original 5-km group used by the benchmark |

Category legends, acquisition periods, and source-specific units should be taken from the manuscript and source metadata. This distribution does not invent a new category legend or infer acquisition dates from the extracted values. Values are preserved from the corrected analysis input; no re-extraction from TIFFs is required.

## Other inputs

| File | Purpose |
|---|---|
| `02_Background_candidates.csv.gz` | Attributed candidate negative locations, including identifiers and distance to the inventory; needed to redraw negatives under the seven exclusion radii |
| `03_Main_fold_assignments.csv.gz` | Fixed main-experiment folds, seeds, observation IDs, and validation groups |
| `04_Clustering_fold_assignments.csv.gz` | Fixed spatial- and attribute-sweep memberships: `sample_id`, `experiment`, `seed`, `fold`; contains no predictions or residuals |
| `05_Inventory_cluster_centres.csv` | Original cluster statistics, including `CL_ID`, `X_CENT`, and `Y_CENT`, used to reconstruct the original group geometry for scale sensitivity |
| `06_Simulation_footprint.csv` | Only `group_x`, `group_y`, and `spatial_block` from the reference point table; defines the 69-cell synthetic domain, not observed responses or trained predictions |
| `07_Landslide_ID_map.csv` | Link between retained positive records and the supplied inventory identifiers |

The fixed folds and cluster geometry are required design inputs, not fitted-model outputs. Compressed CSV files are read directly by pandas; manual decompression is unnecessary.

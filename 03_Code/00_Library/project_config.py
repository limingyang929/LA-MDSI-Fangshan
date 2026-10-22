"""All inputs are resolved relative to the downloaded repository."""
from pathlib import Path
import os
import hashlib
import json
import sys
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[option] = '1'
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / '01_Data'
CONFIG = PROJECT / '02_Config'
CODE = Path(__file__).resolve().parent
output_setting = Path(os.environ.get('LA_MDSI_OUTPUT_DIR', '05_Outputs')).expanduser()
OUTPUT_BASE = output_setting if output_setting.is_absolute() else PROJECT / output_setting
SMOKE = os.environ.get('LA_MDSI_SMOKE') == '1'
OUTPUT = OUTPUT_BASE / ('02_Smoke' if SMOKE else '01_Full')
BENCHMARK = OUTPUT / '01_Benchmark'
EXTENSIONS = OUTPUT / '02_Empirical_extensions'
SIMULATION = OUTPUT / '03_Sampling_simulation'
REPLACEMENT = OUTPUT / '04_Simulation_controls'
LEGACY = OUTPUT / '05_Legacy_baseline'
DEPENDENCE = OUTPUT / '06_Dependence_sensitivity'
INPUT = DATA / '01_Fangshan_samples.csv'
FOLD_SOURCE = DATA / '03_Main_fold_assignments.csv.gz'
CLUSTER_FOLDS = DATA / '04_Clustering_fold_assignments.csv.gz'
COMMON16 = ['Aspect','Curvature','Elevation','TWI','Slope','Landcover','Soil','Distance_to_river','Distance_to_road','FVC','NDVI_recent','NDVI_mean','NDVI_std','NDVI_trend','precipitation_mean','precipitation_max']
EXTRA4 = ['Lithology','Distance_to_fault','Distance_to_coal_mine','Distance_to_seismic_points']
SEEDS = list(range(42,133,10))
FEATURES = {16:[f for f in COMMON16 if f != 'Aspect']+['Aspect_sin','Aspect_cos'],
            20:[f for f in COMMON16+EXTRA4 if f != 'Aspect']+['Aspect_sin','Aspect_cos']}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()
def ensure_output(path):
    path = Path(path)
    if not path.resolve().is_relative_to(OUTPUT_BASE.resolve()):
        raise ValueError('Output must stay inside LA_MDSI_OUTPUT_DIR')
    path.mkdir(parents=True, exist_ok=True)
    return path

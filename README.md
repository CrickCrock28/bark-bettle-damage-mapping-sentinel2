# Bark-Beetle Damage Mapping with Sentinel-2

Pixel-wise **segmentation** and **change detection** on **Sentinel-2** imagery to map **bark-beetle damage** using a **ResNet-50** pretrained on BigEarthNet v2.0.

Developed as a **thesis project** on bark-beetle damage mapping from Sentinel-2.
Thesis (separate repository, text in Italian): [thesis-bark-bettle-damage-mapping-sentinel2](https://github.com/CrickCrock28/thesis-bark-bettle-damage-mapping-sentinel2)

---

## What it does (at a glance)

- **Segmentation (per date):** classifies each pixel as `0 = healthy` or `1 = damage` and exports a GeoTIFF.
- **Change Detection (between two dates):** detects new/expanded damage by comparing model outputs across dates with **Euclidean** or **SAM** distance and **Otsu** thresholding.
- Works on **10 selected Sentinel-2 bands**, using **5×5 patches (radius r=2)** internally to predict per-pixel labels (stitched back to full scene).

---

## Input

- **Sentinel-2 multispectral scenes** (included in this repository under `data/.../sentinel_2/`).
- **Binary ground-truth masks** for training/testing (`data/masks/...`).
- **Optional forest mask** (`data/label_forest_2022/`) to restrict processing to forest pixels.
- (Internal) The pipeline builds **5×5 patches** over the **selected 10 bands**.

> All **paths, file name patterns, selected bands, and hyperparameters** are configured via YAML files in `config/`. If you rename folders/files or tweak training/testing, update them there.


---

## Output

- **Damage segmentation per scene** (GeoTIFF, 1 band): `results/test/pred_{id}.tif`
- **Change-detection map** (GeoTIFF, 1 band): `results/damage_detection/cd_{id}.tif`
- **Metrics report** (per-image precision/recall/F1): `results/metrics.xlsx`
- **(Optional) Training artifacts** (best weights, logs): `results/models/...`

---

## How it works (high level)

1. **Preprocess:** reads scenes, optionally applies the forest mask, and creates normalized **5×5** patches.
2. **Train:** loads **ResNet-50 (BigEarthNet v2.0)**, replaces the head with **2 classes (healthy vs damaged)**, and fine-tunes.
3. **Test:** stitches patch-level predictions back into a full-scene **segmentation GeoTIFF** and computes metrics.
4. **Change Detection:** runs the model on both dates, computes **Euclidean**/**SAM** distance on class probabilities, and applies **Otsu** to create a binary change map.


---

## Project structure

~~~
.
├─ main.py
├─ requirements.txt
├─ config/
|  └─ config_loader.py
├─ data/
|  ├─ dataset.py
|  ├─ preprocess.py
|  ├─ 2019-09-01 - 2019-09-30/sentinel_2/
|  ├─ 2020-09-01 - 2020-09-30/sentinel_2/
|  ├─ masks/
|  │  ├─ test/
|  │  └─ train/
|  └─ label_forest_2022/
├─ model/
|  ├─ BigEarthNetv2_0_ImageClassifier.py
|  ├─ model_tester.py
|  ├─ model.py
|  ├─ pipeline.py
|  ├─ trainer.py
|  ├─ utils.py
|  └─ resnet50-s2-v0.2.0/
├─ results/
└─ scripts/

~~~

---

## Requirements & setup

- Python **3.10+**
- PyTorch (CPU build or a CUDA build matching your GPU)

~~~
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
~~~


---

## Run & CLI (flexible workflow)

You can run **any stage independently** — the pipeline **does not require a fixed order**. Choose only what you need:

- **Preprocess** *(optional)* — create/refresh patch files (use when data layout changes or new scenes are added).
  ~~~
  python main.py --config config/config.yaml --preprocess
  ~~~
- **Train** *(optional)* — fine-tune the 2-class model (healthy vs damaged). Skip if you already have a suitable checkpoint.
  ~~~
  python main.py --config config/config.yaml --train
  ~~~
- **Test** *(optional)* — generate per-scene segmentation GeoTIFFs and compute metrics (requires a model checkpoint).
  ~~~
  python main.py --config config/config.yaml --test
  ~~~
- **Change Detection** *(can be run on its own)* — produce change maps directly from the two configured dates.
  ~~~
  # Euclidean distance on probabilities
  python main.py --config config/damage_detection/euclidean.yaml --damage_detection

  # Spectral Angle Mapper
  python main.py --config config/damage_detection/sam.yaml --damage_detection
  ~~~

**CLI flags (summary)**
- `--config <PATH>`: YAML to control **paths**, **file patterns**, **band order**, and **hyperparameters**.
- `--preprocess`: build patch files from Sentinel-2 scenes (uses forest mask if present).
- `--train`: fine-tune the 2-class model (healthy vs bark-beetle).
- `--test`: reconstruct segmentation GeoTIFFs and compute metrics.
- `--damage_detection`: generate change maps between the two configured dates using the YAML metric (`sam` or `euclidean`) + Otsu.
- `--help`: show all options.


---

## Windows one-click (.bat)

A **batch file** in `scripts/` runs the full pipeline automatically (**preprocess → train → test → change detection**).  
Update the `--config` path near the top if needed, then launch:

~~~
scripts\run_all.bat
~~~

---

## Parameters & configuration

- Control  training hyperparameters (batch size, learning rate, epochs, etc.), paths, file name patterns, selected bands, and training/testing settings via the YAML files in `config/`.
- Change patch size via `radius` (e.g., `r=2 → 5×5`).
- Switch change-detection metric via `testing.distance_metric` (`sam` or `euclidean`).
- If you reorganize data folders, update `paths` and `filenames` accordingly.

---
# MIDAS: Micro-Electron Diffraction Integrated Data Automation System

An automated pipeline for processing Micro-Electron Diffraction (microED)
datasets. The workflow completely handles everything from raw image files
through to an optimally merged final dataset, without requiring user
intervention.

---

## What This Does

MicroED is a method that uses electron diffraction to determine the atomic
structure of molecules when the crystals are too small to study with
traditional X-ray crystallography. Each crystal can only be measured briefly
before being destroyed, and the accessible angular range is limited by the
stage configuration in the transmission electron microscope (TEM), so data
from many crystals must be combined. This pipeline automates the entire
process:

1. Renames and backs up raw diffraction image files
2. Generates XDS configuration files tailored to each dataset
3. Runs XDS to find the crystal lattice and measure reflection intensities
4. Retries indexing automatically with relaxed parameters if the first attempt fails
5. Detects and excludes bad frames and ice rings automatically
6. Extracts quality statistics (completeness, Rmeas, I/sigma, CC½)
7. Applies automated resolution cutoffs based on signal quality
8. Uses a neural network to independently check each dataset's quality
9. Identifies the dominant crystal form and filters incompatible datasets
10. Merges all compatible datasets with XSCALE
11. Finds the optimal combination of datasets to maximise completeness
    while minimising degradation of data quality

---

## Files

| File | Description |
|------|-------------|
| `xds_pipeline.py` | Main pipeline — runs everything end to end |
| `microed_cnn.py` | Neural network module — model, preprocessing, inference |
| `train_cnn.py` | Training script — teaches the CNN using XDS results |
| `structure_solution.py` | Structure solution — POINTLESS, SHELXT, SHELXL |
| `microed_cnn_weights.pt` | Trained model weights (created by `train_cnn.py`) |
| `cell_parameters_summary.csv` | Output spreadsheet generated after each run |
| `TUTORIAL.md` | Step-by-step SOP with acetaminophen worked example |
| `LICENSE` | MIT License |

---

## Requirements

### Python packages

```bash
pip install fabio torch torchvision numpy
```

### XDS

XDS must be installed and on your PATH. Download from https://xds.mr.mpg.de

```bash
cd ~
wget https://xds.mr.mpg.de/XDS-gfortran_Linux_x86_64.tar.gz
tar -xzf XDS-gfortran_Linux_x86_64.tar.gz

# Add to PATH (add to ~/.bashrc to make permanent)
export PATH=~/XDS-gfortran_Linux_x86_64:$PATH

# Verify installation
which xds_par
xds_par | head -4
```

---

## Quick Start

```bash
# Step 1: Process all crystals through XDS and merge with XSCALE
python xds_pipeline.py --folder /path/to/your/data

# Step 2: Train the CNN on your results
python train_cnn.py

# Step 3: Re-run with CNN quality scores feeding into dataset selection
python xds_pipeline.py --folder /path/to/your/data
```

See `TUTORIAL.md` for a complete step-by-step walkthrough with the
acetaminophen test dataset.

---

## Running the Pipeline

```bash
python xds_pipeline.py
```

When prompted, drag your parent folder (the one containing all your crystal
subdirectories) into the terminal and press Enter.

Your folder structure should look like this:

```
parent_folder/
    crystal-1/
        crystal-1_0001.img
        crystal-1_0002.img
        ...
    crystal-2/
        crystal-2_0001.img
        ...
```

### Command line options

```bash
# Pass the folder path directly (skips the interactive prompt)
python xds_pipeline.py --folder /path/to/your/data

# Process multiple crystals in parallel (recommended: 4-8 on a server)
python xds_pipeline.py --folder /path/to/your/data --workers 4

# Watch mode: keep running and process new crystals as they appear
python xds_pipeline.py --folder /path/to/your/data --watch

# Run in the background so the job continues after you close the terminal
nohup python xds_pipeline.py --folder /path/to/your/data --workers 4 > pipeline.log 2>&1 &

# Monitor a background run
tail -f pipeline.log
```

---

## Output

After a successful run, your data folder will contain:

```
parent_folder/
    cell_parameters_summary.csv        # Quality metrics for every crystal
    crystal-1/
        XDS.INP                        # XDS configuration (auto-generated)
        XDS_ASCII.HKL                  # Processed reflections
        CORRECT.LP                     # XDS statistics log
        IDXREF.LP                      # Indexing log
        log/
            crystal-1_XDS_idxref.log
            crystal-1_XDS_integrate.log
    xscale/
        all_compatible/
            XSCALE.HKL                 # All compatible datasets merged (baseline)
        optimal/
            XSCALE.HKL                 # Best subset merged (use this)
        trials/                        # One folder per greedy search step
```

**The file to use for structure determination is `xscale/optimal/XSCALE.HKL`.**

---

## CSV Column Reference

| Column | Source | Description |
|--------|--------|-------------|
| `subdirectory` | pipeline | Crystal folder name |
| `space_group` | CORRECT.LP | Space group number determined by XDS |
| `a` `b` `c` | CORRECT.LP | Unit cell lengths in Angstroms |
| `alpha` `beta` `gamma` | CORRECT.LP | Unit cell angles in degrees |
| `has_hkl` | pipeline | YES if XDS produced a usable HKL file |
| `idxref_indexed` | IDXREF.LP | Number of spots successfully indexed |
| `idxref_total` | IDXREF.LP | Total spots found |
| `idxref_fraction` | IDXREF.LP | Fraction of spots indexed (0 to 1) |
| `completeness_overall` | CORRECT.LP | Overall data completeness (%) |
| `rmeas_overall` | CORRECT.LP | Rmeas — internal consistency (lower is better) |
| `isigi_overall` | CORRECT.LP | Mean I/sigma — signal strength (higher is better) |
| `cc_half_overall` | CORRECT.LP | CC½ — statistical reliability (closer to 1 is better) |
| `resolution_high` | CORRECT.LP | High-resolution limit achieved (Angstroms) |
| `cnn_quality_score` | CNN | CNN quality score (0 = poor, 1 = excellent) |
| `cnn_disagreement` | pipeline | YES if CNN and XDS disagree on the unit cell |

---

## How Dataset Selection Works

**Filter 1 — Dominant crystal form:** For each unit cell parameter the pipeline
computes the median and median absolute deviation (MAD). Any dataset more than
3 MADs from the median is automatically rejected as a wrong indexing solution
or incompatible crystal form.

**Filter 2 — Quality ranking:** Compatible datasets are ranked by a composite
quality score (indexed fraction, completeness, I/sigma, CNN score).

**Greedy forward selection:** Starting with the best dataset, each remaining
dataset is tested by running XSCALE with it included. It is kept only if the
merged completeness, CC½, and Rmeas all genuinely improve.

---

## How the Neural Network Works

`microed_cnn.py` contains a ResNet-18 based convolutional neural network
adapted for single-channel grayscale diffraction images with two outputs:

**Unit cell prediction** — independently predicts the six cell parameters from
raw diffraction images and compares against XDS. Disagreements of more than
5 Å or 5° flag the dataset for review.

**Quality scoring** — predicts a 0–1 quality score from image features (spot
sharpness, signal-to-background). Run `train_cnn.py` after initial pipeline
processing to train it on your own data.

---

## Instrument Configuration

The default XDS.INP template is configured for the UCSC cryo-EM facility
using a ThermoFisher CETA 16M detector operated in 2×2 binning mode.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DETECTOR` | ADSC | XDS detector type identifier |
| `NX` / `NY` | 2048 / 2048 | Detector size in pixels (2×2 binned from 4096×4096) |
| `QX` / `QY` | 0.028 mm | Effective pixel size (14 µm native × 2 binning) |
| `ORGX` / `ORGY` | 1043 / 1046 | Beam centre in pixels (instrument-specific, X ≠ Y) |
| `DETECTOR_DISTANCE` | 1304.07 mm | Sample to detector distance |
| `X-RAY_WAVELENGTH` | 0.025082 Å | Electron wavelength at 200 kV |
| `GAIN` | 15 | Detector gain |
| `OVERLOAD` | 65000 | Pixel saturation threshold |
| `OSCILLATION_RANGE` | 1.0 deg | Rotation per frame |

To adapt for a different instrument, update `_XDS_INP_TEMPLATE`,
`_ORGX_DEFAULT`, and `_ORGY_DEFAULT` in `xds_pipeline.py`.

---

## Troubleshooting

**XDS not on PATH:** `export PATH=~/XDS-gfortran_Linux_x86_64:$PATH`

**XDS license expired:** Download latest from https://xds.mr.mpg.de

**ILLEGAL KEYWORD error:** Delete old XDS.INP files and re-run:
```bash
find /path/to/data -name "XDS.INP" -delete && python xds_pipeline.py
```

**0 datasets processed:** Check `.img` files exist:
```bash
find /path/to/data -name "*.img" | head -5
```

**-99.9% Rmeas:** Each reflection measured only once (multiplicity = 1).
Normal for individual microED datasets — XSCALE merging builds redundancy.

**CNN not loading:** Run `train_cnn.py` first. Pipeline still runs without it.

---

## License Distributed under the MIT License. See `LICENSE` for more information.

MIT License — Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

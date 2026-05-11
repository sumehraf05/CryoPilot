# Tutorial: MicroED Data Processing with Acetaminophen

This tutorial walks through the complete MicroED processing pipeline using
acetaminophen as a test case. Acetaminophen (paracetamol, C₈H₉NO₂) is an
ideal test molecule because its structure is small and well-known, making
it straightforward to verify that the pipeline is working correctly.

**What you will learn:**
- How to organise and prepare raw microED data
- How to run each step of the pipeline and what to expect
- How to interpret the output statistics at each stage
- How to run structure solution and verify the result

---

## Table of Contents

1. [Background: What is microED?](#1-background)
2. [Dataset organisation](#2-dataset-organisation)
3. [Step 1: Environment setup](#3-step-1-environment-setup)
4. [Step 2: Running XDS processing](#4-step-2-running-xds-processing)
5. [Step 3: Understanding the quality report](#5-step-3-understanding-the-quality-report)
6. [Step 4: Training the CNN](#6-step-4-training-the-cnn)
7. [Step 5: Structure solution](#7-step-5-structure-solution)
8. [Expected results for acetaminophen](#8-expected-results-for-acetaminophen)
9. [Interpreting statistics](#9-interpreting-statistics)
10. [Adapting to your own data](#10-adapting-to-your-own-data)

---

## 1. Background

### What is microED?

Micro-Electron Diffraction (microED) is a cryo-electron microscopy technique
for determining atomic-resolution structures of materials that are difficult
or impossible to study by traditional X-ray crystallography. Unlike X-ray
methods which require crystals of at least 10–100 µm, microED works with
crystals as small as 100 nm.

The key challenge of microED is that each crystal can only be measured over
a limited angular range before the electron beam destroys it through radiation
damage. Additionally, the mechanical tilt range of the TEM stage is typically
±70°, which limits how much of reciprocal space can be sampled from any single
crystal. As a result, a single crystal usually provides incomplete data
(20–40% completeness is typical), and data from many crystals must be combined
to build a complete dataset.

### What is XDS?

XDS (X-ray Detector Software) is the industry-standard software for processing
rotation diffraction data. Despite its name, it handles electron diffraction
data well when configured appropriately. XDS performs:

- **COLSPOT / IDXREF:** Finds strong spots and indexes them to determine the
  crystal orientation and unit cell
- **INTEGRATE:** Measures the intensity of every reflection across all frames
- **CORRECT:** Applies corrections (absorption, decay, geometry) and computes
  final statistics

### What is XSCALE?

XSCALE is the companion program to XDS for merging data from multiple crystals.
It scales all datasets to a common reference frame and produces a single merged
reflection file. The pipeline runs XSCALE automatically after XDS finishes.

---

## 2. Dataset Organisation

Your raw data should be organised with one subdirectory per crystal:

```
my_experiment/
    crystal-1/
        crystal-1_0001.img     ← frame 1
        crystal-1_0002.img     ← frame 2
        ...
        crystal-1_0080.img     ← frame 80
    crystal-2/
        crystal-2_0001.img
        ...
    crystal-3/
        ...
```

The pipeline will automatically rename files if they do not start at frame 1.
Original files are always backed up to `crystal-N/backup/` before renaming.

**Naming:** Subdirectory names can be anything (`crystal-1`, `xtal_a`, `run01`,
etc.). The pipeline discovers them automatically by searching for directories
containing `.img` files.

**Acetaminophen test data:** The test dataset used during development of this
pipeline consists of 16 crystals collected at 200 kV on a ThermoFisher CETA
16M detector. After processing, crystals 13, 14, and 15 were found to provide
the best data quality and produced the optimal merged dataset.

---

## 3. Step 1: Environment Setup

### Install Python dependencies

```bash
pip install fabio torch torchvision numpy
```

Verify:
```bash
python3 -c "import fabio, torch, numpy; print('All packages OK')"
```

### Install and verify XDS

```bash
cd ~
wget https://xds.mr.mpg.de/XDS-gfortran_Linux_x86_64.tar.gz
tar -xzf XDS-gfortran_Linux_x86_64.tar.gz
export PATH=~/XDS-gfortran_Linux_x86_64:$PATH
```

Verify the license is current:
```bash
xds_par | head -5
# Should show: "Copy licensed until DD-Mon-YYYY"
# If you see "license expired" -- download a new copy from xds.mr.mpg.de
```

Make the PATH permanent:
```bash
echo 'export PATH=~/XDS-gfortran_Linux_x86_64:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Set PyTorch cache location

On shared cluster systems, set the PyTorch cache to your home directory to
avoid permission errors when downloading pre-trained weights:

```bash
echo 'export TORCH_HOME=~/.torch' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Step 2: Running XDS Processing

Run the pipeline and point it at your data folder:

```bash
python xds_pipeline.py
```

```
Drag the PARENT folder here, then press Enter:
> /path/to/my_experiment
```

Or pass the folder directly:

```bash
python xds_pipeline.py --folder /path/to/my_experiment
```

### What to expect during processing

For each crystal you will see output like this:

```
=== Processing: /path/to/my_experiment/crystal-14 ===
  dist=1304.07mm  wl=0.025082A  ORGX=1043.0  ORGY=1046.0
  XDS.INP generated.
  Images: 80 files  first=crystal-14_00001.img  (8.4 MB)
  Running Phase 1 (XYCORR INIT COLSPOT IDXREF)...
  Phase 1 done in 47.3 s  (exit 0)
  IDXREF: 232/411 indexed (56.4%)  cell=9.44 9.44 15.22 89.7 89.8 59.6
  Running Phase 2 (DEFPIX INTEGRATE CORRECT)...
  Phase 2 done in 38.1 s  (exit 0)
  SG=1  Cell: 9.443 9.435 15.225 89.684 89.752 59.622
  Overall: comp=35.6%  Rmeas=0.000  I/sig=0.0  CC½=0.000
  Dataset done in 87.4 s  HKL=YES
```

**Key things to watch:**
- `IDXREF: 232/411 indexed (56.4%)` — 56% of spots were indexed, which is
  reasonable. If this is below 15%, the pipeline will automatically retry with
  relaxed parameters.
- `HKL=YES` — the crystal was successfully processed and produced output data
- `comp=35.6%` — individual completeness is expected to be 20–40% per crystal

Crystals that completely fail to index will show `HKL=NO` and are
automatically excluded from merging.

### Runtime

Each crystal takes 1–5 minutes depending on the number of frames and the
server speed. For 16 crystals, expect 20–60 minutes total.

To use multiple CPU cores and process crystals in parallel:

```bash
python xds_pipeline.py --folder /path/to/my_experiment --workers 4
```

---

## 5. Step 3: Understanding the Quality Report

After all crystals are processed, the pipeline prints a dataset quality report
followed by the XSCALE merging results.

### Dataset Quality Report

```
========================================================================
  DATASET QUALITY REPORT  (11 datasets with HKL files)
========================================================================
  Dataset               Idx%   Comp%   I/sig      a      c
------------------------------------------------------------------------
  crystal-14            56.4   35.6     0.0   9.443  15.225
  crystal-13            54.0   34.7     0.0   9.347  15.514
  crystal-20            95.3   34.4     0.0   9.332  15.277
  crystal-15            44.2   32.4     0.0   9.387  15.232
  crystal-16            18.8   24.2     0.0   9.441  15.316
  crystal-17            27.5   18.8     0.0  10.691  74.798  ← wrong cell
  crystal-12            31.3   33.0     0.0  14.389  16.008  ← wrong cell
```

- **Idx%** — percentage of spots XDS successfully indexed. Higher is better.
  Below 15% usually means the crystal was too damaged or misaligned.
- **Comp%** — individual completeness. 20–40% is normal per crystal.
- **a, c** — unit cell dimensions. Datasets with very different values (like
  crystal-17 with c=74.8 Å vs the typical 15.3 Å) are wrong indexing solutions.

### Filter 1: Dominant crystal form

```
  FILTER 1: Removing datasets with incompatible unit cells
  Target cell:  a=9.44  b=9.44  c=15.32  alpha=89.7  beta=89.8  gamma=60.6
  EXCLUDED  crystal-17  Wrong cell -- c: value=74.80  deviation=595 MADs
  EXCLUDED  crystal-12  Wrong cell -- gamma: value=108.21  deviation=59 MADs
  KEPT      crystal-13  Cell matches consensus
  KEPT      crystal-14  Cell matches consensus
  ...
  Result: 5 kept, 6 excluded
```

The pipeline automatically identifies which datasets have the correct unit
cell and which are wrong. In the acetaminophen test dataset, 6 out of 11
datasets had wrong unit cells and were automatically excluded.

### Filter 2: Quality ranking

```
  FILTER 2: Ranking compatible datasets by quality
  #1  score=0.142  idx=56.4%  comp=35.6%  crystal-14
  #2  score=0.139  idx=54.0%  comp=34.7%  crystal-13
  #3  score=0.138  idx=95.3%  comp=34.4%  crystal-20
  #4  score=0.130  idx=44.2%  comp=32.4%  crystal-15
  #5  score=0.097  idx=18.8%  comp=24.2%  crystal-16
```

### XSCALE merging results

```
  MERGE A: All 5 compatible datasets
  comp=84.0%  Rmeas=0.461  CC1/2=0.605  I/sig=1.5  n_unique=4497

  MERGE B: Optimal subset (greedy search)
  + ACCEPTED crystal-13  score=0.625  comp=63.1%  Rmeas=0.183  CC½=0.966
  - REJECTED crystal-20  (trial score 0.520 < best 0.625)
  + ACCEPTED crystal-15  score=0.635  comp=71.3%  Rmeas=0.229  CC½=0.923
  - REJECTED crystal-16  (trial score 0.568 < best 0.635)

  Optimal (3): comp=71.3%  Rmeas=0.229  CC1/2=0.924  I/sig=1.9  n_unique=3817
```

In this example, the optimal merge uses 3 crystals rather than all 5 because
crystals 20 and 16, despite having compatible unit cells, degraded the merged
data quality when included. This is the correct behaviour — the pipeline chose
quality over quantity.

**The file to use for everything downstream is:**
```
/path/to/my_experiment/xscale/optimal/XSCALE.HKL
```

---

## 6. Step 4: Training the CNN

After the pipeline has processed your data, train the neural network on your
results:

```bash
python train_cnn.py
```

```
Drag the PARENT folder here:
> /path/to/my_experiment
```

Training reads `cell_parameters_summary.csv` and uses:
- XDS-measured cell parameters as the regression target
- Indexed fraction as the quality label (0 = failed, 1 = fully indexed)

Training takes a few minutes. When finished, `microed_cnn_weights.pt` is
saved to your data folder.

Then re-run the pipeline to use the trained CNN weights:

```bash
python xds_pipeline.py --folder /path/to/my_experiment
```

On this second run, the CNN will add quality scores to every dataset that
feed into the XSCALE subset selection.

---

## 7. Step 5: Structure Solution

With the merged XSCALE.HKL in hand, run structure solution:

```bash
python structure_solution.py
```

```
Drag the PARENT folder:
> /path/to/my_experiment

Molecular formula (default: C8 H9 N1 O2)
Press Enter for default:
>
```

The script runs:

1. **POINTLESS** — determines the true space group by analysing systematic
   absences. For acetaminophen this typically finds space group P-1 (triclinic,
   space group #2).

2. **SHELXT** — structure solution by dual-space direct methods. Reads the
   merged reflection data and finds the positions of all atoms in the unit
   cell without any prior structural knowledge.

3. **SHELXL** — least-squares refinement. Takes the SHELXT solution and
   iteratively adjusts atomic positions and displacement parameters to best
   fit the observed diffraction data.

Expected output for acetaminophen:
```
  FINAL RESULTS
  Space group  : P -1  (#2)
  Unit cell    : 9.39 9.44 15.23 90.25 90.07 119.81
  SHELXT  R1   : 0.21  (20 atoms placed)
  SHELXL  R1   : 0.18   wR2 : 0.41
  Quality: OK  (R1 < 0.20, further refinement may help)
```

The R1 factor measures the agreement between observed and calculated
diffraction intensities. For microED data at this stage:
- R1 < 0.15 is excellent
- R1 < 0.25 is good for a starting point
- R1 > 0.30 suggests the solution may need refinement or the space group
  may need to be reconsidered

**Note for publication:** The SHELXL refinement in this script uses X-ray
scattering factors as a starting point. For publication-quality microED
structures, electron scattering factors must be used. Consult your advisor
about the `ABIN` keyword in SHELXL for electron diffraction refinement.

---

## 8. Expected Results for Acetaminophen

### Unit cell

Acetaminophen (Form I) has a triclinic unit cell:

| Parameter | Expected | Typical range from microED |
|-----------|----------|---------------------------|
| a | 9.40 Å | 9.3–9.5 Å |
| b | 9.44 Å | 9.3–9.5 Å |
| c | 15.23 Å | 15.0–15.5 Å |
| α | 90.0° | 89–91° |
| β | 90.0° | 89–91° |
| γ | 119.8° | 119–121° |
| Space group | P-1 (#2) | |

### Merged data quality targets

| Statistic | Acceptable | Good |
|-----------|-----------|------|
| Completeness | > 60% | > 75% |
| Rmeas | < 0.35 | < 0.20 |
| CC½ | > 0.80 | > 0.90 |
| I/sigma | > 1.0 | > 2.0 |

### Structure solution targets

| Statistic | Acceptable | Good |
|-----------|-----------|------|
| SHELXT R1 | < 0.30 | < 0.20 |
| SHELXL R1 | < 0.25 | < 0.15 |
| SHELXL wR2 | < 0.60 | < 0.40 |

### Acetaminophen atomic structure

Acetaminophen (paracetamol) has the molecular formula C₈H₉NO₂ with a
molecular weight of 151.16 Da. The structure consists of a para-substituted
benzene ring with a hydroxyl group (–OH) and an acetamide group (–NHCOCH₃).

The known reference structure is deposited in the Cambridge Structural
Database as entry **HXACAN**. After a successful pipeline run and structure
solution, the refined coordinates should match this reference with an RMSD
of less than 0.05 Å for non-hydrogen atoms.

To verify your result against the reference:
1. Open the refined `.res` file from `structure_solution/3_shelxl/` in
   Mercury (part of CCP4) or ORTEP
2. Compare visually with HXACAN from the Cambridge Structural Database
3. The two structures should be chemically identical — benzene ring, hydroxyl
   group, and acetamide group all in the correct connectivity

---

## 9. Interpreting Statistics

### Indexed fraction (Idx%)

The fraction of diffraction spots that XDS successfully assigned to lattice
positions.

- **> 50%:** Good crystal, well-indexed
- **15–50%:** Acceptable, the pipeline may retry with relaxed parameters
- **< 15%:** Poor crystal — possible causes: too much radiation damage, crystal
  out of eucentric height, or wrong detector parameters

### Completeness

What fraction of all theoretically possible unique reflections were measured.

- **Individual crystal:** 20–40% is normal and expected for microED
- **Merged dataset:** > 60% is the minimum useful threshold; > 80% is good

Low completeness in the merged dataset means you need more crystals.

### Rmeas (redundancy-independent merging R-factor)

Measures how consistently the same reflection was measured when it appeared
multiple times (in different crystals). Lower is better.

- **< 0.20:** Excellent internal consistency
- **0.20–0.35:** Good
- **0.35–0.50:** Acceptable for a starting structure
- **> 0.50:** High — may indicate non-isomorphous crystals being merged together

If Rmeas shows as 0.000, it means every reflection was only measured once
(multiplicity = 1) so consistency cannot be calculated. This is normal for
individual crystals but should resolve after merging multiple datasets.

### CC½ (half-dataset correlation coefficient)

The correlation between two half-datasets formed by randomly splitting the
data. This measures statistical reliability without requiring a reference.

- **> 0.95:** Excellent
- **0.90–0.95:** Good
- **0.80–0.90:** Acceptable
- **< 0.80:** Borderline — the data may have significant systematic errors

### I/sigma (mean intensity / mean uncertainty)

The signal-to-noise ratio averaged over all reflections.

- **> 5:** Strong signal throughout
- **2–5:** Acceptable
- **1–2:** Weak signal — structure solution may work but refinement will be
  challenging
- **< 1:** Signal is not distinguishable from noise at this resolution

---

## 10. Adapting to Your Own Data

### Different molecule

Change the molecular formula when prompted by `structure_solution.py`. For
example for lysozyme or another protein, you would give the formula of the
asymmetric unit contents.

### Different instrument

The XDS.INP template contains instrument-specific parameters. To adapt for
a different detector or microscope:

1. Open `xds_pipeline.py` in a text editor
2. Find `_XDS_INP_TEMPLATE` near the top of the file
3. Update the following parameters to match your instrument:
   - `NX`, `NY` — detector size in pixels
   - `QX`, `QY` — pixel size in mm
   - `ORGX`, `ORGY` — beam centre position in pixels
   - `DETECTOR_DISTANCE` — if you want to hardcode a default
   - `X-RAY_WAVELENGTH` — electron wavelength for your accelerating voltage
   - `OVERLOAD` — detector saturation value
4. Update `_ORGX_DEFAULT` and `_ORGY_DEFAULT` near the `generate_xds_inp`
   function to match your beam centre

Common voltage / wavelength values:

| Voltage | Wavelength |
|---------|-----------|
| 80 kV  | 0.04176 Å |
| 120 kV | 0.03349 Å |
| 200 kV | 0.02508 Å |
| 300 kV | 0.01969 Å |

### Different number of frames

If your crystals have more or fewer than 80 frames, the pipeline handles
this automatically — it reads the number of frames from the actual image
files and sets DATA_RANGE accordingly.

### Watch mode for live data collection

During a data collection session, run the pipeline in watch mode so it
processes crystals as they are collected:

```bash
python xds_pipeline.py --folder /path/to/my_experiment --watch
```

The pipeline checks for new crystal subdirectories every 60 seconds and
processes them automatically. Stop with Ctrl+C when collection is complete.

---

## Quick Reference: Complete Command Sequence

```bash
# Setup (one time)
export PATH=~/XDS-gfortran_Linux_x86_64:$PATH
export TORCH_HOME=~/.torch

# Process crystals through XDS and XSCALE
python xds_pipeline.py --folder /path/to/my_experiment

# Train CNN on results
python train_cnn.py
# (enter /path/to/my_experiment when prompted)

# Re-run with CNN
python xds_pipeline.py --folder /path/to/my_experiment

# Structure solution
python structure_solution.py
# (enter /path/to/my_experiment and formula C8 H9 N1 O2 when prompted)

# Key output files:
# /path/to/my_experiment/cell_parameters_summary.csv  -- quality metrics
# /path/to/my_experiment/xscale/optimal/XSCALE.HKL   -- merged data
# /path/to/my_experiment/structure_solution/3_shelxl/refined.res  -- structure
```

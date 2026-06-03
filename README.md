# Breast Density Classification on VinDr-Mammo with PyTorch

## Overview

This project aims to develop a reproducible **deep learning baseline** for breast density classification from mammography images using **PyTorch**.

The first objective is to classify breast density into two clinically meaningful groups:

- **Low density**: BI-RADS A/B
- **High density**: BI-RADS C/D

The second objective is to extend the task to the four ordered BI-RADS breast density categories:

- **BI-RADS A**
- **BI-RADS B**
- **BI-RADS C**
- **BI-RADS D**

This project is designed as a first step toward **medical imaging AI**, **ordinal-aware learning**, and **multimodal representation learning** for breast cancer decision support.

## Clinical and Scientific Motivation

Breast density is an important imaging biomarker in mammography. Dense breast tissue can reduce mammographic sensitivity and may complicate lesion detection. It is also clinically relevant for breast cancer screening and risk assessment.

From a machine learning perspective, breast density classification is a relevant task because BI-RADS density categories are not independent classes: they follow an ordinal structure from **A to D**. A confusion between **A and B** is less severe than a confusion between **A and D**.

This makes the task suitable for exploring:

- medical image classification;
- class imbalance handling;
- clinically relevant evaluation metrics;
- ordinal-aware learning;
- model interpretability;
- future multimodal learning with imaging and clinical/tabular data.

## Dataset

The project is intended to use the **VinDr-Mammo** dataset, a publicly available mammography dataset containing breast-level information including BI-RADS assessment and breast density labels.

The data will not be included directly in this repository. Users should download the dataset from the official source and place it locally in the `data/` directory.

Expected local structure:

```txt
data/
├── raw/
├── processed/
├── dummy_images/
└── metadata.csv
```

## Methodological Approach

This project follows a lightweight **V-model inspired methodology**, adapted to a research-oriented medical AI project.

The goal is to ensure traceability between:

1. clinical need;
2. system requirements;
3. software requirements;
4. modular implementation;
5. unit verification;
6. integration verification;
7. final validation and reporting.

This approach is used to make the project more structured, reproducible and aligned with good engineering practices in biomedical research.

## Planned Pipeline

The planned pipeline is:

```txt
Metadata CSV
→ Image loading
→ Image preprocessing
→ PyTorch Dataset
→ CNN baseline model
→ Training
→ Evaluation
→ Results report
```

The first baseline will use:

- **Framework**: PyTorch
- **Model**: ResNet18
- **Input size**: 224×224 initially
- **Task 1**: binary classification A/B vs C/D
- **Task 2**: four-class classification A/B/C/D
- **Metrics**:
  - accuracy;
  - macro-F1;
  - balanced accuracy;
  - confusion matrix.

## Repository Structure

```txt
vindr-mammo-density-classification/
├── configs/
│   └── resnet18_density.yaml
├── data/
├── docs/
│   ├── 01_clinical_need.md
│   ├── 02_system_requirements.md
│   ├── 03_software_requirements.md
│   ├── 04_verification_plan.md
│   └── 05_validation_plan.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── reports/
│   └── results.md
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Current Status

The project is currently in its initial setup phase.

Completed:

- project structure;
- Python virtual environment;
- installation of core packages;
- JupyterLab setup;
- initial V-model documentation structure;
- environment verification with PyTorch CPU.

Next steps:

- create a dummy dataset;
- implement the PyTorch Dataset class;
- test image and label loading;
- implement a ResNet18 baseline;
- run a first training loop on dummy data;
- prepare the real VinDr-Mammo metadata pipeline.

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

On Windows cmd:

```bash
venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the installation:

```python
import torch
import torchvision
import timm
import sklearn
import cv2

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
```

A CPU-only PyTorch installation is sufficient for local development and testing. Heavier training can later be performed on Google Colab or another GPU environment.

## Development Roadmap

### Phase 1 — Project setup

- Create project structure
- Set up Python environment
- Create initial documentation
- Connect repository to GitHub

### Phase 2 — Dummy pipeline

- Generate dummy images
- Create dummy metadata CSV
- Implement PyTorch Dataset
- Verify image and label loading
- Test model output shape

### Phase 3 — Baseline model

- Implement ResNet18 baseline
- Train on a small subset
- Compute first metrics
- Save initial results

### Phase 4 — VinDr-Mammo integration

- Download and organize VinDr-Mammo
- Prepare metadata CSV
- Map BI-RADS density labels
- Implement real image preprocessing
- Train binary density classifier

### Phase 5 — Evaluation and reporting

- Evaluate accuracy, macro-F1 and balanced accuracy
- Generate confusion matrix
- Analyze errors
- Document limitations

### Phase 6 — Extensions

Possible extensions include:

- four-class BI-RADS density classification;
- ordinal-aware loss functions;
- Grad-CAM visualizations;
- class imbalance strategies;
- integration of clinical/tabular variables.

## Limitations

This project is not intended to be a clinical-grade diagnostic system.

It is a research and portfolio project designed to:

- build a reproducible medical imaging AI pipeline;
- explore breast density classification;
- demonstrate applied PyTorch skills;
- prepare future work on multimodal medical AI.

Important limitations to document include:

- dataset imbalance;
- image preprocessing choices;
- reduced image resolution;
- lack of external validation;
- annotation variability;
- difference between a proof of concept and a clinically validated system.

## Author

**Goulwen Le Bras**  
Biomedical engineer interested in medical imaging, scientific software development, biomedical R&D and artificial intelligence applied to healthcare.
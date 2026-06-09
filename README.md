# Breast Density Classification on VinDr-Mammo with PyTorch

## 1. Overview

This repository is a portfolio and learning project focused on **breast density classification from mammography images** using **PyTorch**.

The first target task is a binary classification of BI-RADS breast density:

- **low density**: BI-RADS density A/B;
- **high density**: BI-RADS density C/D.

A later extension may address the four-class ordinal task:

- BI-RADS density A;
- BI-RADS density B;
- BI-RADS density C;
- BI-RADS density D.

The project is developed progressively: first with a fully tested dummy dataset, then with real VinDr-Mammo DICOM integration, and finally with stronger training, evaluation and reporting.

## 2. Motivation

This project was motivated by my interest in a PhD proposal on **Multimodal Representation Learning for Breast Cancer Decision Support from Imaging and Clinical Data**. The proposal focuses on the integration of medical imaging and structured clinical data, with particular attention to breast cancer, multimodal representation learning, self-supervised learning, real-world data imperfections, and ordinal clinical targets such as BI-RADS categories.

Breast density classification on VinDr-Mammo is used here as a focused and reproducible first step toward these topics. It allows the project to start from a clear medical imaging task before later extensions toward ordinal-aware learning, tabular metadata integration, and multimodal image + clinical data representation learning.

Breast density is an important imaging characteristic in mammography. Dense breast tissue can make image interpretation more difficult and is relevant in breast cancer screening and risk assessment. From a machine-learning perspective, breast density is also an interesting target because the BI-RADS density categories A, B, C and D are **ordered** rather than independent labels.

This project is designed to strengthen practical skills in:

- medical image loading and preprocessing;
- DICOM handling;
- PyTorch Dataset and DataLoader design;
- CNN-based classification;
- model evaluation with clinically relevant metrics;
- reproducible engineering practices for medical AI.

## 3. Dataset

The intended dataset is **VinDr-Mammo**, officially available through **PhysioNet**:

> Pham, H. H., Nguyen Trung, H., & Nguyen, H. Q. (2022). *VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography* (version 1.0.0). PhysioNet. https://doi.org/10.13026/br2v-7517

The associated Scientific Data paper describes VinDr-Mammo as a dataset of **5,000 full-field digital mammography exams**, with four standard views per exam, breast-level assessment, breast density labels, and finding-level annotations.

The data are **not stored in this GitHub repository**. Users must obtain the dataset from the official source and respect the dataset license and access conditions.

Expected external data structure:

```txt
E:/physionet.org/files/vindr-mammo/1.0.0/
├── metadata.csv
├── breast-level_annotations.csv
├── finding_annotations.csv
└── images/
    ├── <study_id>/
    │   ├── <image_id>.dicom
    │   ├── <image_id>.dicom
    │   ├── <image_id>.dicom
    │   └── <image_id>.dicom
    └── ...
```

For the current project, the main file is:

```txt
breast-level_annotations.csv
```

Important columns for the first baseline:

```txt
study_id
image_id
laterality
view_position
height
width
breast_birads
breast_density
split
```

The target label is:

```txt
breast_density
```

The binary mapping is:

```txt
A/B -> 0  low density
C/D -> 1  high density
```

## 4. Methodological Approach

This project follows a lightweight **V-model inspired methodology**, adapted to a research and portfolio project in biomedical AI.

The goal is to keep traceability between:

1. clinical need;
2. data and task definition;
3. software requirements;
4. modular implementation;
5. unit testing;
6. integration testing;
7. model evaluation;
8. reporting and limitations.

This helps separate clinical reasoning, data preparation, software implementation and experimental validation.

## 5. Workflow

The project is developed in two progressive stages.

### Stage 1 — Dummy pipeline

The first stage validates the complete software pipeline on synthetic data:

```txt
dummy metadata CSV
→ dummy images
→ PyTorch Dataset
→ torchvision transforms
→ DataLoader
→ ResNet18 model
→ training loop
→ evaluation metrics
→ end-to-end dummy training script
```

This stage does not aim to produce medically meaningful results. Its purpose is to verify that the codebase is modular, testable and functional before introducing real medical images.

### Stage 2 — VinDr-Mammo integration

The second stage will connect the existing pipeline to real VinDr-Mammo DICOM images:

```txt
breast-level_annotations.csv
→ image path creation from study_id and image_id
→ breast density label mapping
→ DICOM image loading
→ PyTorch Dataset
→ DataLoader
→ ResNet18 baseline
→ training and validation
→ results report
```

The VinDr-Mammo `split` column will be used carefully. The official test split should remain the test set, while the official training split may be divided into training and validation subsets.

## 6. Repository Structure

Current and planned structure:

```txt
vindr-mammo-density-classification/
├── configs/
├── data/
│   └── README.md
├── docs/
├── notebooks/
├── reports/
│   └── results.md
├── src/
│   ├── create_dummy_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_dummy_training.py
│   ├── dicom_utils.py                  # planned: DICOM loading
│   ├── prepare_vindr_metadata.py       # planned: VinDr metadata adapter
│   ├── run_vindr_smoke_test.py         # planned: real-data smoke test
│   └── train_vindr.py                  # planned: stronger training script
├── tests/
│   ├── test_dataset.py
│   ├── test_dataloader.py
│   ├── test_model.py
│   ├── test_training_step.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_dicom_utils.py             # planned
│   └── test_prepare_vindr_metadata.py  # planned
├── .gitignore
├── README.md
└── requirements.txt
```

## 7. Current Implementation

The following components are already implemented:

- project folder structure;
- Python virtual environment;
- PyTorch CPU environment verification;
- `.gitignore` configuration to exclude data, virtual environments and model checkpoints;
- synthetic dummy image generation;
- dummy metadata CSV generation;
- `MammographyDensityDataset`;
- torchvision image transforms;
- DataLoader batch creation;
- ResNet18 baseline model;
- binary and four-class output shape tests;
- minimal training step;
- `train_one_epoch()`;
- `evaluate_model()`;
- evaluation metrics:
  - accuracy;
  - macro-F1;
  - balanced accuracy;
  - confusion matrix;
- end-to-end dummy training script: `src/run_dummy_training.py`;
- initial reporting structure in `reports/results.md`.

## 8. Next Steps

The next development steps are:

- add DICOM image loading with `pydicom`;
- adapt the Dataset to support `.dicom` and `.dcm` files;
- create a VinDr-Mammo metadata adapter;
- generate image paths from `study_id` and `image_id`;
- map `breast_density` to binary labels:
  - A/B → 0;
  - C/D → 1;
- use the official VinDr-Mammo `split` column properly;
- run a real-data smoke test on a small subset;
- run a first mini-training experiment on VinDr-Mammo;
- save metrics and confusion matrix;
- update `reports/results.md` with real experiment results;
- improve documentation and limitations.

## 9. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows cmd:

```bash
venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the future DICOM stage, `pydicom` is required:

```bash
pip install pydicom
```

Check PyTorch:

```python
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
```

Local CPU execution is sufficient for code development, unit tests and smoke tests. Heavier training should be performed on a GPU environment such as Google Colab, Kaggle, a lab server or another remote GPU machine.

## 10. Usage

### Run all tests

```bash
pytest
```

### Run the current dummy pipeline

```bash
python src\run_dummy_training.py
```

This validates the software pipeline, but the numerical results are not medically meaningful because the images are synthetic.

### Planned VinDr-Mammo commands

After downloading VinDr-Mammo outside the repository, the planned metadata preparation command is:

```bash
python src\prepare_vindr_metadata.py
```

The planned real-data smoke test command is:

```bash
python src\run_vindr_smoke_test.py
```

The smoke test will check that:

- real DICOM files are found;
- DICOM images can be loaded;
- labels are correctly mapped;
- the Dataset and DataLoader work;
- the model can process real batches;
- one short training/evaluation cycle runs without crashing.

## 11. Evaluation

The first baseline will report:

- accuracy;
- macro-F1;
- balanced accuracy;
- confusion matrix.

Balanced accuracy and macro-F1 are especially important because medical datasets can be imbalanced. Accuracy alone can be misleading if one class is over-represented.

For a stronger version, the project may add:

- AUC for binary A/B vs C/D classification;
- per-view analysis by CC and MLO;
- per-laterality analysis;
- comparison between binary and four-class classification;
- ordinal-aware evaluation.

## 12. Data and Git Policy

The repository must not contain:

- VinDr-Mammo DICOM files;
- raw medical images;
- generated local metadata with absolute paths;
- model checkpoints;
- virtual environments.

Recommended `.gitignore` entries:

```gitignore
data/*
!data/README.md

models/
*.pth
*.pt

venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
```

## 13. Limitations

This repository is not a clinical-grade diagnostic system.

Current limitations:

- the implemented pipeline has so far been validated on dummy data;
- real DICOM integration is still to be completed;
- no clinically meaningful model performance has been established yet;
- no external validation has been performed;
- image preprocessing choices are still preliminary;
- the first baseline uses reduced image resolution;
- class imbalance has not yet been fully handled;
- interpretability methods are not yet implemented.

The project should be understood as a reproducible engineering and learning baseline for medical imaging AI, not as a validated medical device or clinical decision support system.

## 14. Future Work

Possible extensions include:

- four-class BI-RADS density classification;
- ordinal-aware loss functions;
- class imbalance strategies;
- saved training curves and confusion matrix figures;
- Grad-CAM visualizations;
- multimodal image + tabular learning using VinDr-Mammo metadata;
- self-supervised or representation learning experiments;
- external validation or cross-dataset analysis if data access allows.

In relation to the PhD topic, the longer-term direction would be to move from a supervised image-only baseline toward richer patient representations that combine mammography images with structured metadata or clinical variables.

## 15. References

1. PhD proposal: *Multimodal Representation Learning for Breast Cancer Decision Support from Imaging and Clinical Data*.
2. Pham, H. H., Nguyen Trung, H., & Nguyen, H. Q. (2022). *VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography* (version 1.0.0). PhysioNet. https://doi.org/10.13026/br2v-7517
3. Nguyen, H. T. et al. (2023). *VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography*. Scientific Data. https://doi.org/10.1038/s41597-023-02100-7

## 16. Author

**Goulwen Le Bras**  
Biomedical engineer interested in medical imaging, scientific software development, biomedical R&D and artificial intelligence applied to healthcare.

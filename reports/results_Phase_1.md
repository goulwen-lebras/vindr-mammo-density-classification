\# Results



\## Current stage



The current version of the project validates an end-to-end deep learning pipeline on a synthetic dummy dataset.



The goal of this stage is not to obtain meaningful medical performance, but to verify that all software components work together correctly.



\## Dummy dataset



A synthetic dummy dataset was generated to test the pipeline.



It contains:



20 synthetic grayscale images

4 BI-RADS-like density labels: A, B, C, D

binary mapping:

A/B → 0

C/D → 1



The images are randomly generated and do not contain medical information.



\## Implemented components



src/create\_dummy\_dataset.py

src/dataset.py

src/model.py

src/train.py

src/evaluate.py

src/run\_dummy\_training.py



\## Tests



The following tests were implemented with pytest:



dataset loading

image and label retrieval

image transforms

DataLoader batch creation

ResNet18 output shape

minimal training step

train loop

evaluation metrics



\## Metrics



The current evaluation metrics are computed on synthetic random images.



Therefore, the numerical values are not clinically meaningful.



The implemented metrics are:



accuracy

macro-F1

balanced accuracy

confusion matrix



\## Limitations



This stage only validates the software pipeline.



Main limitations:



synthetic random images

no real mammography data yet

no clinically meaningful performance

no train/validation/test split yet

no class imbalance handling yet

no external validation



\## Next steps



The next stage will focus on preparing real VinDr-Mammo metadata.



Planned steps:



inspect VinDr-Mammo metadata structure;

identify breast density labels;

map BI-RADS density A/B/C/D;

create binary labels A/B vs C/D;

create clean metadata CSV;

prepare train/validation/test split;

connect the real metadata CSV to the existing PyTorch Dataset.


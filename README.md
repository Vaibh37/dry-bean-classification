**Dry Bean Classification**

**Project Overview**

This project implements a supervised multiclass classification workflow for identifying seven dry bean varieties from sixteen numerical descriptors of their geometry and shape. The objective is to develop, validate, and interpret a Decision Tree model using a clearly defined experimental process.

The classifier operates on measurements extracted from bean images. It does not process raw images or perform image recognition.

**Dataset and Preparation**

The Dry Bean dataset is sourced from the UCI Machine Learning Repository and contains 13,611 observations across seven classes. Each observation contains sixteen numerical features and one categorical class label.

Data inspection identified no missing values and 68 exact duplicate records. Removing those duplicates reduced the dataset to 13,543 observations. The cleaned data was divided using a stratified 80 percent training and 20 percent testing split, resulting in 10,834 training observations and 2,709 testing observations. The split uses a fixed random seed of 42.

Class frequencies were examined before modeling because the dataset is not uniformly distributed across varieties.

**Modeling and Selection**

A DecisionTreeClassifier with entropy as the splitting criterion serves as the primary estimator. An initial tree constrained to a maximum depth of three was evaluated as a baseline.

Hyperparameter selection was performed on the training partition with GridSearchCV and five fold cross validation. The search evaluated twenty combinations of maximum tree depth and minimum samples per leaf, using classification accuracy as the selection metric.

The selected configuration permits a maximum depth of eight and requires at least five training samples per leaf. GridSearchCV then refitted the selected estimator on the complete training partition.

**Evaluation Results**

Baseline mean cross validation accuracy: **80.46%**

Selected model mean cross validation accuracy: **91.07%**

Selected model training accuracy: **93.88%**

Final test accuracy: **90.85%**

Macro averaged F1 score: **0.92**

Weighted F1 score: **0.91**

The difference between cross validation accuracy and final test accuracy is small in this experiment. These results describe performance on the supplied dataset and split, rather than a guarantee of performance on independently collected data.

**Error Analysis**

Per class metrics and the confusion matrix reveal differences that overall accuracy does not capture. BOMBAY achieved perfect precision and recall across its 104 test observations. SIRA had the lowest recall at approximately 0.83, with 68 actual SIRA observations predicted as DERMASON.

The uneven class distribution makes per class precision, recall, and F1 scores important alongside aggregate accuracy.

**Model Interpretation**

Impurity based feature importance identifies MajorAxisLength and MinorAxisLength as the largest contributors to the fitted tree, with importance scores of approximately 0.320 and 0.211. ShapeFactor3 and ShapeFactor1 follow at approximately 0.147 and 0.137.

These values describe the contribution of each feature to impurity reduction in this estimator. They should not be interpreted as causal effects or as evidence that features with low importance contain no predictive information.

The implementation generates a class distribution plot, confusion matrix visualization, sorted feature importance chart, and a partial Decision Tree diagram using Matplotlib.

**Implementation and Execution**

The project is implemented in Python using pandas for dataset operations, scikit learn for model training and evaluation, Matplotlib for visualization, and openpyxl for Excel input.

Install the dependencies listed in `requirements.txt`, retain the dataset workbook in the `data` directory, and run `python main.py` from the repository root. The script executes the training and evaluation workflow and displays the visualizations.

**Limitations**

The experiment evaluates a single model family using one stratified dataset split. No external dataset validation, deployment interface, or raw image inference pipeline is included. Additional benchmarking and validation would be required before considering operational use.

**Data Attribution and Licensing**

Dataset: Dry Bean, UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/602/dry%2Bbean

DOI: https://doi.org/10.24432/C50S4B

The original dataset is distributed under Creative Commons Attribution 4.0 International. The repository source code is distributed under the MIT License. Dataset attribution is also provided in the repository.

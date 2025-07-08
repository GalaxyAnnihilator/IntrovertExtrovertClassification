# Personality Classification from Social Behavior
A machine learning application that decides whether you are introvert or extrovert

## Try It Live

https://huggingface.co/spaces/tmdeptrai3012/PersonalityPrediction

--

## Dataset
Kaggle: https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data

## Exploratory Data Analysis

### Class Distribution
![Class Distribution](./Figures/class_distribution.png)

### Box Plot
![Box Plot](./Figures/box_plots.png)

### Pair Plot of Numeric Features by Personality
![Pair Plot](./Figures/pair_plot.png)

### Correlation Heatmap
![Correlation Heatmap](./Figures/correlation_heatmap.png)

---

## Model Training Results

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Extrovert    | 0.91      | 0.89   | 0.90     | 298     |
| Introvert    | 0.89      | 0.91   | 0.90     | 282     |
| **Accuracy** |           |        | **0.90** | **580** |
| Macro Avg    | 0.90      | 0.90   | 0.90     | 580     |
| Weighted Avg | 0.90      | 0.90   | 0.90     | 580     |

### Confusion Matrix
![Confusion Matrix](./Figures/confusion_matrix.png)

### ROC Curve
![ROC Curve](./Figures/roc_curve.png)

### Feature Importance
![Feature Importance](./Figures/feature_importance_plot.png)

---

## Tech Stack

- Python
- Scikit-learn
- Gradio
- DVC + CML
- GitHub Actions CI/CD


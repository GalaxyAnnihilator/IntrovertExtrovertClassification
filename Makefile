install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black *.py

eda:
	python eda.py

train:
	python train.py

eval:
	cat << EOF > report.md
# Exploratory Data Analysis

## Class Distribution
![Class Distribution](./Figures/class_distribution.png)

## Box Plot
![Blox Plot](./Figures/box_plots.png)

## Pair Plot of Numeric Features by Personality
![Pait Plot](./Figures/pair_plot.png)

## Correlation Heatmap
![Correlation Heatmap](./Figures/correlation_heatmap.png)

# Training Results

## Model Metrics
$$(cat ./Figures/metrics.txt)

## Confusion Matrix Plot
![Confusion Matrix](./Figures/confusion_matrix.png)

## ROC Curve
![ROC Curve](./Figures/roc_curve.png)

## Feature Importance
![Feature Importance](./Figures/feature_importance_plot.png)
EOF

	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload tmdeptrai3012/CICDforMLOps ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload tmdeptrai3012/CICDforMLOps ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload tmdeptrai3012/CICDforMLOps ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
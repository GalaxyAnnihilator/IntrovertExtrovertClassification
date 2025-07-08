install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black *.py

eda:
	python eda.py

train:
	python train.py

eval:
	@echo "# Exploratory Data Analysis" > report.md
	@echo "" >> report.md
	@echo "## Class Distribution" >> report.md
	@echo "![Class Distribution](./Figures/class_distribution.png)" >> report.md
	@echo "" >> report.md
	@echo "## Box Plot" >> report.md
	@echo "![Box Plot](./Figures/box_plots.png)" >> report.md
	@echo "" >> report.md
	@echo "## Pair Plot of Numeric Features by Personality" >> report.md
	@echo "![Pair Plot](./Figures/pair_plot.png)" >> report.md
	@echo "" >> report.md
	@echo "## Correlation Heatmap" >> report.md
	@echo "![Correlation Heatmap](./Figures/correlation_heatmap.png)" >> report.md
	@echo "" >> report.md
	@echo "# Training Results" >> report.md
	@echo "" >> report.md
	@echo "## Model Metrics" >> report.md
	@cat ./Figures/metrics.txt >> report.md
	@echo "" >> report.md
	@echo "## Confusion Matrix Plot" >> report.md
	@echo "![Confusion Matrix](./Figures/confusion_matrix.png)" >> report.md
	@echo "" >> report.md
	@echo "## ROC Curve" >> report.md
	@echo "![ROC Curve](./Figures/roc_curve.png)" >> report.md
	@echo "" >> report.md
	@echo "## Feature Importance" >> report.md
	@echo "![Feature Importance](./Figures/feature_importance_plot.png)" >> report.md

	@cml comment create report.md

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
	huggingface-cli upload tmdeptrai3012/PersonalityPrediction ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload tmdeptrai3012/PersonalityPrediction ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload tmdeptrai3012/PersonalityPrediction ./Figures /Figures --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
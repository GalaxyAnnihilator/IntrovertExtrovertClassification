import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import os

df = pd.read_csv("./Data/personality_dataset.csv")

numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_columns = ['Stage_fear', 'Drained_after_socializing']
target_column = 'Personality'

plt.figure(figsize=(8,6))
sns.color_palette("rocket", as_cmap=True)
sns.countplot(x=target_column,data=df,palette='mako',hue=target_column)
plt.title('Class Distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.savefig('./Figures/class_distribution.png')

plt.figure(figsize=(8,6))
for i,col in enumerate(numeric_columns,1):
    plt.subplot(3,2,i)
    sns.boxplot(x=target_column,y=col,data=df,palette="magma",hue=target_column)
    plt.title(f'{col} by Personality')
plt.tight_layout()
plt.savefig('./Figures/box_plots.png')

sns.pairplot(df[numeric_columns + [target_column]], hue=target_column, diag_kind='hist')
plt.suptitle('Pair Plot of Numeric Features by Personality', y=1.02)
plt.savefig('./Figures/pair_plot.png')

plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_columns].corr(),annot=True,cmap="coolwarm",fmt='.2f',vmin=-1,vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig('./Figures/correlation_heatmap.png')

X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ColumnTransformer to apply them
transform = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_columns),
    ("num", numerical_pipeline, numeric_columns),
])

# Final pipeline
pipe = Pipeline([
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
pipe.fit(X_train, y_train)  

predictions = pipe.predict(X_test)
result = classification_report(y_test,predictions,labels=pipe.classes_)

with open('./Figures/metrics.txt','w') as f:
    f.write(result)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True,fmt='d',cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('./Figures/confusion_matrix.png')


# ROC Curve

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test).ravel()

fpr, tpr, _ = roc_curve(y_test_bin, pipe.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test_bin, pipe.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('./Figures/roc_curve.png')

# Get feature names after preprocessing
cat_features = pipe.named_steps['preprocessing'].transformers_[0][2]
num_features = pipe.named_steps['preprocessing'].transformers_[1][2]

# Get encoded categorical feature names
encoder = pipe.named_steps['preprocessing'].named_transformers_['cat'].named_steps['encoder']
encoded_cat_features = encoder.get_feature_names_out(cat_features)

# Combine all final feature names
final_features = list(encoded_cat_features) + list(num_features)

# Extract feature importances from the model
model = pipe.named_steps['model']
importances = model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance,palette='viridis',hue=feature_importance['Importance'])
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('./Figures/feature_importance_plot.png')  

import skops.io as sio
sio.dump(pipe,"./Model/personality_pipeline.skops")
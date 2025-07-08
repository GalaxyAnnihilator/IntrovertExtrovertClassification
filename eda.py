import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


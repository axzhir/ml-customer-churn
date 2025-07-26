# Machine Learning Project: Decision Tree Classifier on Customer Churn Data

In this machine learning project, we train and evaluate multiple decision tree classifiers on the **cell2cell churn dataset** to predict whether a customer will churn. This project focuses on the modeling phase of the machine learning life cycle and includes data preprocessing, model training, hyperparameter tuning, and evaluation.

## 📌 Objectives

* Load and explore the **cell2cell** dataset
* Prepare data for modeling

  * Handle missing values
  * One-hot encode categorical features
  * Identify labels and features
  * Split the dataset into training and test sets
* Train and evaluate decision tree models using different hyperparameters
* Analyze model performance using accuracy scores and visualizations

## 🧪 Sample Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("cell2cell.csv")  # Replace with correct path or load method

# Define label
label = 'Churn'
df = df.dropna(subset=[label])  # Remove rows with missing label

# One-hot encode categorical variables
df = pd.get_dummies(df)

# Split features and label
X = df.drop(columns=[label])
y = df[label]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two models with different depths
clf1 = DecisionTreeClassifier(max_depth=3)
clf2 = DecisionTreeClassifier(max_depth=10)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predict and evaluate
acc1 = accuracy_score(y_test, clf1.predict(X_test))
acc2 = accuracy_score(y_test, clf2.predict(X_test))

print(f"Accuracy (max_depth=3): {acc1:.2f}")
print(f"Accuracy (max_depth=10): {acc2:.2f}")

# Plot accuracies
plt.bar(['depth=3', 'depth=10'], [acc1, acc2], color=['skyblue', 'lightgreen'])
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
```

## 📊 Analysis

To further improve performance, try:

* Tuning other hyperparameters such as `min_samples_split`, `min_samples_leaf`, and `criterion`
* Visualizing the decision tree to understand feature splits
* Performing feature selection to reduce dimensionality

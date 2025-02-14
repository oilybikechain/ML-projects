# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading models
import os  # For checking file existence

iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame (optional, but makes it easier to view the data)
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) # Add the species names
print(df.head())

# 2. Data Exploration (Optional, but recommended!)
# Get summary statistics
print(df.describe())

# Check the distribution of each feature
df.hist(figsize=(10, 8))  # Creates histograms for each feature
plt.tight_layout()  # Adjusts subplot parameters for a tight layout.
plt.show()

# Visualize the relationships between features (Scatter plots)
sns.pairplot(df, hue='species') # Creates scatterplots for each pair of features, colored by species
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

model = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print a classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix (optional)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
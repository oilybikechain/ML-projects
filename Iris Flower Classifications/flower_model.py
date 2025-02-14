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

# 1. Define a path for saving the model
MODEL_PATH = "iris_model.joblib"

# 2. Load new Data
# Load the Iris dataset (or simulate new data)
iris = load_iris()
X = iris.data
y = iris.target

# Split new data into training, validation, and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=43)
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_train_new, y_train_new, test_size=0.2, random_state=43)

# 3. Check for existence of model and if doesn't exist make a default set to train, if it exists, pull in the best one
if os.path.exists(MODEL_PATH):
    # Load the pre-trained model
    print("Loading pre-trained model...")
    best_model = joblib.load(MODEL_PATH)
else:
    # Train a new model
    print("Training a new model...")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_new = scaler.fit_transform(X_train_new)
    X_val_new = scaler.transform(X_val_new)
    X_test_new = scaler.transform(X_test_new)

    # Choose a Machine Learning Model
    model = DecisionTreeClassifier(random_state=42)

    # Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'max_depth': [None, 2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_new, y_train_new)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

# 4. Load the scaler and append all the features to that scaler.
# Check for existence of model and if it exist, load it
SCALER_PATH = "iris_scaler.joblib"

if os.path.exists(SCALER_PATH):
    # Load the pre-trained model
    print("Loading pre-trained scaler...")
    scaler = joblib.load(SCALER_PATH)
    scaler.fit(X_train_new) # Add on the previous training parameters
    print("Finished training pre-trained scaler")
else:
    # Train a new model
    print("Training a new scaler...")

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train_new)
    #best_model = grid_search.best_estimator_
    print("Finished training new scaler")
joblib.dump(scaler, SCALER_PATH)

#5. Dump all the values into a numpy array for training, validation, and testing

X_train_new = scaler.transform(X_train_new)
X_val_new = scaler.transform(X_val_new)
X_test_new = scaler.transform(X_test_new)

# 6. Retrain the Model using more data
# Retrain the model using all available data
best_model.fit(X_train_new, y_train_new)

# 7. Save the Model
print("Saving the model...")
joblib.dump(best_model, MODEL_PATH)

# 8. Make Predictions and Evaluate (on the NEW test set)
y_pred = best_model.predict(X_test_new)

# Calculate accuracy
accuracy = accuracy_score(y_test_new, y_pred)
print(f"Accuracy: {accuracy}")

# Print a classification report
print(classification_report(y_test_new, y_pred, target_names=iris.target_names))

# Create a confusion matrix
cm = confusion_matrix(y_test_new, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix (optional)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
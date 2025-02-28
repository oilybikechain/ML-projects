# Goal is to predict if someone on the titanic survived. The data  is very messy
# Understand nature of the data .info() .describe()
# Histograms and boxplots 
# Value counts 
# Missing data 
# Correlation between the metrics 
# Explore interesting themes 
    # Wealthy survive? 
    # By location 
    # Age scatterplot with ticket price 
    # Young and wealthy Variable? 
    # Total spent? 
# Feature engineering 
# preprocess data together or use a transformer? 
    # use label for train and test   
# Scaling?

# Model Baseline 
# Model comparison with CV 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the data
df = pd.read_csv("train.csv")

# # 2. Data Exploration and Cleaning
# # Check for missing values
# print(df.isnull().sum())
# print(df.head())
# print(df.info())

# # Find which categories are numbers
# print(df.describe().columns)

# Separate numerical data sets
df_num = df[['Age','SibSp','Parch','Fare']]
df_cat = df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# # Univariate histogram analysis
# for i in df_num.columns:
#     plt.hist(df_num[i])
#     plt.title(i)
#     plt.show()


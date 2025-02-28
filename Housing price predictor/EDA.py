import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.max_columns', 200)
df = pd.read_csv("train.csv")

# print(df.info())
# print(df.shape)
# print(df.head(20))
# print(df.describe())
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# print(numeric_cols)

df_num = df[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]

for i in df_num.columns:
    plt.figure()
    plt.hist(df_num[i])
    plt.title(i)
    plt.xlabel(i) #Added labels for clarity
    plt.ylabel("Frequency") # Added labels for clarity
    plt.show()

#Consider dropping month sold
# Change MSSubClass to one hot encoding
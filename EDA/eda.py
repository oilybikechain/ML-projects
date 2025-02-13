import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('max_columns', 200)

plt.style.use('ggplot')
df = pd.read_csv("coaster_db.csv")

# Shows number of rows and columns
print(df.shape)

# Shows the first x number of rows (default 5)
# need to specify the amount of columns so everything is shown
print(df.head(20))
# pd.set_option('display.max_columns', 200)

print(df.columns)

# prints the data types of the columns
print(df.dtypes)

# shows statistics about the data
print(df.describe())

#Now cleaning data
# Drop irrelevant columns
df = df[['coaster_name','Location', 'Status','Manufacturer','year_introduced','latitude', 'longitude','Type_Main',
       'opening_date_clean','speed_mph', 'height_ft','Inversions_clean', 'Gforce_clean']].copy()

#Example of dropping single column, axis = 1 tells the computer that it is dropping a column 
df.drop(['Opening date'], axis=1)

#force change in type
df['opening_date_clean'] = pd.to_datetime(df['opening_date_clean'])

#Rename
df.rename(columns = {'coaster_name':'Newname'})

#IMPORTANT, sum of missing values
df.isna().sum()

#Find duplicated values
df.loc[df.duplicated()]
df.loc[df.duplicated(subset = ['Newname'])]

#Checking columns
df.query("Coaster_Name == 'Crystal Beach Cyclone'")

#Remove duplicates with certain number of same variables ~ means inverse reset index clears empty rows
df = df.loc[~df.duplicated(subset=['Coaster_Name','Location','Opening_Date'])] \
    .reset_index(drop=True).copy()

#Univariate analysis
#Bar graph of single column
ax = df['Year_Introduced'].value_counts() \
    .head(10) \
    .plot(kind='bar', title='Top 10 Years Coasters Introduced')
ax.set_xlabel('Year Introduced')
ax.set_ylabel('Count')

# Histogram
ax = df['Speed_mph'].plot(kind='hist',
                          bins=20,
                          title='Coaster Speed (mph)')
ax.set_xlabel('Speed (mph)')

# Density plot
ax = df['Speed_mph'].plot(kind='kde',
                          title='Coaster Speed (mph)')
ax.set_xlabel('Speed (mph)')

#Numbers
df['Type_Main'].value_counts()

#Bivariate analysis
#Scatter plot
df.plot(kind='scatter',
        x='Speed_mph',
        y='Height_ft',
        title='Coaster Speed vs. Height')
plt.show()

#multivariate analysis
# More advanced scatterplot using seaborn, including year. e.g. hue of dots determined hue
ax = sns.scatterplot(x='Speed_mph',
                y='Height_ft',
                hue='Year_Introduced',
                data=df)
ax.set_title('Coaster Speed vs. Height')
plt.show()

#Matrix of pairplots
sns.pairplot(df,
             vars=['Year_Introduced','Speed_mph',
                   'Height_ft','Inversions','Gforce'],
            hue='Type_Main')
plt.show()

#numerical correlation between different functions
df_corr = df[['Year_Introduced','Speed_mph',
    'Height_ft','Inversions','Gforce']].dropna().corr()
df_corr
#heatmap of correlation
sns.heatmap(df_corr, annot=True)

#finding specific data
#Group data and find average speed, generating horizontal bar graph
ax = df.query('Location != "Other"') \
    .groupby('Location')['Speed_mph'] \
    .agg(['mean','count']) \
    .query('count >= 10') \
    .sort_values('mean')['mean'] \
    .plot(kind='barh', figsize=(12, 5), title='Average Coast Speed by Location')
ax.set_xlabel('Average Coaster Speed')
plt.show()

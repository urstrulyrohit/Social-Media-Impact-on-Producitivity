#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().system('pip install skimpy')
from skimpy import skim

# Suppress warnings
warnings.filterwarnings("ignore")



# In[2]:


# Loading the dataset
df = pd.read_csv("D:\\Users\\zenit\\Downloads\\Time-Wasters on Social Media.csv")
df.head()


# # **EDA**

# #### Helper functions

# In[3]:


# Function to compute describe and skewness
def describe_and_skew(df, column):
    # Calculate describe() for the specified column
    describe_result = df[[column]].describe()

    # Calculate skewness for the specified column
    skew_result = df[[column]].skew()

    return describe_result, f'Skewness of {skew_result}'


# In[4]:


df = pd.read_csv("D:\\Users\\zenit\\Downloads\\Time-Wasters on Social Media.csv")
df.head()


# In[5]:


#Separate DF variable for ML
df1 = pd.read_csv("D:\\Users\\zenit\\Downloads\\Time-Wasters on Social Media.csv")
df1.head()


# In[6]:


df.shape


# ## Data Cleaning

# In[7]:


missing_data = df.isna().sum()
if not missing_data[missing_data > 0].empty:
    print(missing_data[missing_data > 0])
else:
    print("No missing data found!")


# In[8]:


d_info= pd.DataFrame(df.info())
print(d_info)


# In[9]:


d_unique = pd.DataFrame(df.nunique())
d_unique


# #### Our data seeems to be okay

# ## Basic Statistics for Numerical Features

# In[10]:


df.describe()


# In[11]:


df.columns


# ## Univariate Analysis

# #### Distribution of Age

# In[12]:


describe_and_skew(df, 'Age')


# In[13]:


plt.figure(figsize=(12,
                    6))
ax=sns.histplot(df['Age'], bins=30, kde=True)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black')
plt.title('Age Distribution')
plt.show()


# #### Distribution of Location

# In[14]:


plt.figure(figsize=(12,
                    6))
ax = sns.countplot(data=df, x='Location')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black')
plt.title('Location Distribution')
plt.show()


# In[15]:


df['Location']


# #### Distribution of Gender

# In[16]:


plt.figure(figsize=(10,6))
ax = sns.countplot(x='Gender', data=df)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black')
plt.title('Gender Distribution')
plt.show()


# #### Income distribution

# In[17]:


describe_and_skew(df, 'Income')


# In[18]:


plt.figure(figsize=(12,
                    6))
ax = sns.histplot(data=df, x='Income', bins=30,  kde=True)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='maroon')
plt.title('Income Distribution')
plt.show()


# #### Debt distribution

# In[19]:


plt.figure(figsize=(12,
                    6))
ax = sns.countplot(data=df, x='Debt', color='lightgray')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='maroon')
plt.title('Income Distribution')
# sns.swarmplot(x=df['Debt'], color='blue', alpha=0.5)
plt.title('Distribution of Debt', fontsize=16)
plt.xlabel('Debt', fontsize=14)
plt.xticks(fontsize=12)

plt.show()


# #### Distribution of Profession

# In[20]:


plt.figure(figsize=(12,
                    6))
ax = sns.countplot(data=df, x='Profession')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Profession Distribution')
plt.show()


# #### Distribution of Platform

# In[21]:


plt.figure(figsize=(12,
                    6))
ax = sns.countplot(data=df, x='Platform')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Platform Distribution')
plt.show()


# #### Distribution of Total Time Spent

# In[22]:


describe_and_skew(df, 'Total Time Spent')


# In[23]:


plt.figure(figsize=(12,6))
ax = sns.histplot(data=df, x='Total Time Spent', bins=30, kde=True)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Total Time Spent Distribution')
plt.show()


# ## Distribution of Video Category

# In[24]:


plt.figure(figsize=(12,
                    6))
ax = sns.countplot(data=df, x='Video Category')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Video Category Distribution')
plt.show()


# ## Distribution of Video Length

# In[25]:


plt.figure(figsize=(12,
                    6))
ax = sns.histplot(data=df, x='Video Length', bins=30, kde=True)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Video Length Distribution')
plt.show()


# ## Distribution of Engagement

# In[26]:


describe_and_skew(df, 'Engagement')


# In[27]:


plt.figure(figsize=(12,
                    6))
ax = sns.histplot(data=df, x='Engagement', bins=30, kde=True)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Engagement Distribution')
plt.show()


# #### Distribution of Watch Reason

# In[28]:


# Example data: Assume 'Gender' has been extracted as a Series
watch_reason_counts = df['Watch Reason'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(watch_reason_counts, labels=watch_reason_counts.index, autopct='%1.1f%%', startangle=140)

# Adding a title
plt.title('Watch Reason Distribution')

# Display the plot
plt.show()


# #### Distribution of Device Type

# In[29]:


# Example data: Assume 'Gender' has been extracted as a Series
device_type_counts = df['DeviceType'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(device_type_counts, labels=device_type_counts.index, autopct='%1.1f%%', startangle=140)

# Adding a title
plt.title('DeviceType Distribution')

# Display the plot
plt.show()


# #### Distribution of CurrentActivity

# In[30]:


# Example data: Assume df['CurrentActivity'] contains the activity categories
plt.figure(figsize=(18, 13))

# Create a count plot with horizontal bars
ax = sns.countplot(y='CurrentActivity', data=df, order=df['CurrentActivity'].value_counts().index, palette='coolwarm')

# Calculate total number of observations
total = len(df['CurrentActivity'])

# Annotate each bar with the corresponding count and percentage
for p in ax.patches:
    count = p.get_width()
    percentage = 100 * count / total
    ax.annotate(f'{count:.0f} ({percentage:.1f}%)', (count + 0.3, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=15, color='black')

# Add a title
plt.title('Distribution of Current Activities with Counts and Percentages', fontsize=16)
plt.xlabel('X-axis', fontsize=15)
plt.ylabel('Y-axis', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Show the plot
plt.show()


# #### Distribution of CurrentType

# In[31]:


import pandas as pd
import matplotlib.pyplot as plt

# Univariate analysis for ConnectionType
connection_counts = df['ConnectionType'].value_counts()

# Create a bar chart
plt.figure(figsize=(8, 6))
connection_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Connection Type')
plt.ylabel('Count')
plt.title('Distribution of Connection Types')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# #### Distribution of CurrentType

# In[32]:


connection_counts = df['ConnectionType'].value_counts().reset_index()
connection_counts.columns = ['ConnectionType', 'Count']

fig = px.treemap(connection_counts,
                 path=['ConnectionType'],
                 values='Count',
                 title='Distribution of Connection Types')

fig.update_traces(textinfo="label+value+percent parent")
fig.show()


# ## Distribution of ProductivityLoss

# In[34]:


describe_and_skew(df, 'ProductivityLoss')


# In[33]:


plt.figure(figsize=(12,6))
ax = sns.histplot(data=df, x='ProductivityLoss', bins=30, kde=True)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('ProductivityLoss Distribution')
plt.show()


# In[35]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=df, x='ProductivityLoss')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('ProductivityLoss Distribution')
plt.show()


# In[36]:


df['ProductivityLoss'].unique()


# ## Distribution of Satisfaction

# In[37]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=df, x='Satisfaction')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Satisfaction Distribution')
plt.show()


# In[ ]:





# ## Distribution of Watch Time

# In[38]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=df, x='Watch Time')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Watch Time Distribution')
plt.show()


# ## Distribution of Addiction Level

# In[39]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=df, x='Addiction Level')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Addiction Level Distribution')
plt.show()


# ## Distribution of Frequency

# In[40]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=df, x='Frequency')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')
plt.title('Frequency Distribution')
plt.show()


# ## Bivariate Analysis

# #### Gender Vs Demographics

# In[41]:


# Creating the contingency table with counts
contingency_table = pd.crosstab(df['Gender'], df['Demographics'], margins=True, margins_name="Total")

# Creating the contingency table with percentages
contingency_percentage = pd.crosstab(df['Gender'], df['Demographics'], normalize='all', margins=True, margins_name="Total") * 100

# Combining counts and percentages into a single table
combined_table = contingency_table.astype(str) + " (" + contingency_percentage.round(2).astype(str) + "%)"

# Converting to a DataFrame and applying visual aesthetics
styled_table = combined_table.style.set_caption("Contingency Table (Crosstab) Gender and Demographics") \
                                    .background_gradient(cmap="YlGnBu", axis=None) \
                                    .set_properties(**{'text-align': 'center', 'border': '1px solid black'}) \
                                    .set_table_styles([{'selector': 'caption', 'props': [('text-align', 'center'), ('font-size', '16px')]}])

# Display the styled table
styled_table


# #### Enagement by platform

# In[42]:


# Bar Plot of Average Engagement
plt.figure(figsize=(12, 6))
ax = df.groupby('Platform')['Engagement'].mean().sort_values(ascending=False).plot(kind='bar', color='skyblue')
for p in ax.patches:
    ax.annotate(f'{round(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='baseline', fontsize=15, color='black')

plt.title('Average Engagement by Platform', fontsize=16)
plt.xlabel('Platform', fontsize=14)
plt.ylabel('Average Engagement', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# 6. Statistical Summary
df.groupby('Platform')['Engagement'].describe()


# #### Age by Video Category Preference

# In[43]:


age_bins = pd.cut(df['Age'], bins=5)
heatmap_data = pd.crosstab(age_bins, df['Video Category'], normalize='index')

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Age Distribution Across Video Categories', fontsize=16)
plt.xlabel('Video Category', fontsize=12)
plt.ylabel('Age Bins', fontsize=12)
plt.tight_layout()
plt.show()


# #### Satisfactor vs ProductivityLoss

# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Satisfaction', y='ProductivityLoss')
plt.title('Satisfaction vs Productivity Loss', fontsize=16)
plt.xlabel('Satisfaction', fontsize=12)
plt.ylabel('Productivity Loss', fontsize=12)
plt.tight_layout()
plt.show()


# In[45]:


correlation = df['Satisfaction'].corr(df['ProductivityLoss'])
print(f"Correlation between Satisfaction and Productivity Loss: {correlation:.2f}")


# #### Video Category vs Satisfaction

# Total Time Spent vs ProductivityLoss

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic Statistics
print(df.groupby('Video Category')['Satisfaction'].describe())

# 2. Bar Plot of Average Satisfaction by Video Category
avg_satisfaction = df.groupby('Video Category')['Satisfaction'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
avg_satisfaction.plot(kind='bar')
plt.title('Average Satisfaction by Video Category', fontsize=16)
plt.xlabel('Video Category', fontsize=12)
plt.ylabel('Average Satisfaction', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ## Multivariate Analysis

# #### Age vs Income vs Profession vs Self Control vs Satisfaction vs ProductivityLoss

# In[47]:


import pandas as pd
import numpy as np

variables = ['Age','Income','Self Control', 'Satisfaction', 'ProductivityLoss']
df1[variables].describe()

# Check for correlations
correlation_matrix = df1[variables].corr()
print("\nCorrelation Matrix:")
correlation_matrix


# In[48]:


sns.pairplot(df1[variables])
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()


# In[49]:


# Identify categorical columns
categorical_columns = ['Gender', 'Location', 'Profession', 'Demographics', 'Platform',
                       'Satisfaction', 'Video Category','Frequency', 'Watch Reason', 'DeviceType', 'OS', 'Watch Time',
                       'CurrentActivity', 'ConnectionType']

# Apply Label Encoding to categorical columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df1[column] = le.fit_transform(df1[column])
    label_encoders[column] = le


# #### ConnectionType vs CurrentActivity

# In[50]:


# Grouping and plotting
professions_by_gender = df.groupby("ConnectionType")["CurrentActivity"].value_counts().unstack(fill_value=0)
ax = professions_by_gender.plot(kind="bar", stacked=True)

# Adding labels on the bars
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=10, color='black')

# Adding labels and title
plt.xlabel("Connection Type")
plt.ylabel("Current Activity")
plt.title("Connection Type by Current Activity")
plt.xticks(rotation=0)
plt.show()


# In[51]:


# Creating the contingency table with counts
contingency_table = pd.crosstab(df['ConnectionType'], df['CurrentActivity'], margins=True, margins_name="Total")

# Creating the contingency table with percentages
contingency_percentage = pd.crosstab(df['ConnectionType'], df['CurrentActivity'], normalize='all', margins=True, margins_name="Total") * 100

# Combining counts and percentages into a single table
combined_table = contingency_table.astype(str) + " (" + contingency_percentage.round(2).astype(str) + "%)"

# Converting to a DataFrame and applying visual aesthetics
styled_table = combined_table.style.set_caption("Contingency Table (Crosstab) Gender and Demographics") \
                                    .background_gradient(cmap="YlGnBu", axis=None) \
                                    .set_properties(**{'text-align': 'center', 'border': '1px solid black'}) \
                                    .set_table_styles([{'selector': 'caption', 'props': [('text-align', 'center'), ('font-size', '16px')]}])

# Display the styled table
styled_table


# # **PCA**

# In[52]:


# Loading the dataset
data = pd.read_csv("D:\\Users\\zenit\\Downloads\\Time-Wasters on Social Media.csv")


# In[53]:


# Identify numerical and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object', 'bool']).columns

# Define a ColumnTransformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Apply the transformations
data_processed = preprocessor.fit_transform(data)

# Check the shape and type of the processed data
data_processed_shape = data_processed.shape
type_data_processed = type(data_processed)

data_processed_shape, type_data_processed


# In[54]:


# Perform PCA
pca = PCA()
pca.fit(data_processed)

# Get the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Create a DataFrame to display the explained variance of each component
pca_results = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Explained Variance Ratio': explained_variance_ratio,
    'Cumulative Variance Ratio': cumulative_variance_ratio
})

pca_results.head(10)  # Displaying the first 10 principal components


# In[55]:


# Get the loadings (components) from the PCA object
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=numeric_features.tolist() + list(preprocessor.named_transformers_['cat'].get_feature_names_out()))

# Display loadings for the first few principal components
print("Loadings for the first few principal components:")
print(loadings.head())


# Plotting cumulative variance explained for all components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()


# In[56]:


# Extract the variable names (numeric + one-hot encoded categorical features)
variable_names = numeric_features.tolist() + list(preprocessor.named_transformers_['cat'].get_feature_names_out())

# Create a DataFrame for the loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=variable_names)

# Display the loadings as a table
loadings


# In[57]:


pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]

# Create a DataFrame to display them
pc1_pc2_loadings = pd.DataFrame({
    'Variable': variable_names,
    'PC1 Loadings': pc1_loadings,
    'PC2 Loadings': pc2_loadings
})

# Print the loadings for PC1 and PC2
print(pc1_pc2_loadings)


# # **Logistic Regression**

# In[58]:


UsageLevel_classes = df['Usage Level Buckets'].unique()
target_names = list(UsageLevel_classes)
target_names


# In[59]:


# Apply Label Encoding to categorical columns
# Identify categorical columns

categorical_columns = df.drop('Usage Level Buckets', axis=1)
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


# In[60]:


# Defining X and Y variables
x = df.drop('Usage Level Buckets', axis=1).to_numpy()
y = df['Usage Level Buckets'].to_numpy()

# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Scaling the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[61]:


# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train2, y_train)

# Predictions
y_pred = log_reg.predict(x_test2)


# In[62]:


# Evaluation Report and Matrix
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=target_names))



# In[63]:


# Boxplot Visualization
df1 = df.drop('Usage Level Buckets', axis=1)
corr = df1.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(df1.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df1.columns)
ax.set_yticklabels(df1.columns)
plt.show()


# # **SVM**

# In[64]:


x = df.drop('Usage Level Buckets', axis=1).to_numpy()
y = df['Usage Level Buckets'].to_numpy()

# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Scaling the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[65]:


# Script for SVM
from sklearn.svm import SVC
svm = SVC(random_state=100)
svm.fit(x_train2, y_train)
predictions = svm.predict(x_test2)

# Evaluation Report and Matrix
print('\nSVM Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\nSVM Classification Report:')
print(classification_report(y_test, predictions, target_names=target_names))


# In[66]:


# Define the parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Initialize the SVM classifier
svm = SVC()

# Initialize GridSearchCV
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, n_jobs=-1, cv=3)

# Fit the model
grid_search_svm.fit(x_train2, y_train)

# Print the best parameters and score for SVM
print("Best parameters found for SVM: ", grid_search_svm.best_params_)
print("Best cross-validation score for SVM: ", grid_search_svm.best_score_)

# Use the best estimator to make predictions on test data
best_svm = grid_search_svm.best_estimator_

# Predict on the test set
y_pred_svm = best_svm.predict(x_test2)

# Evaluation Report and Matrix for SVM
print('\n')
print('Confusion Matrix for SVM:')
print(confusion_matrix(y_test, y_pred_svm))
print('\n')
print('Classification Report for SVM:')
print(classification_report(y_test, y_pred_svm, target_names=target_names))


# # **NN-MLP**

# In[67]:


#Script for Neural Network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes = (5, 4, 5),
    activation = 'relu',
    solver = 'adam',
    max_iter = 10000,
    random_state = 100
)


mlp.fit(x_train2, y_train)
predictions = mlp.predict(x_test2)

#Evaluation Report and Matrix
from sklearn.metrics import confusion_matrix, classification_report
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\n')
print('Classification Report:')
print(classification_report(y_test, predictions, target_names=target_names)
)
print(len(df))


# ### **Tuning our model (with Grid Search)**

# In[68]:


# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Initialize the classifier with increased max_iter and reduced tol
mlp = MLPClassifier(max_iter=1000, tol=1e-4, n_iter_no_change=10)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)

# Fit the model
grid_search.fit(x_train2, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)


# In[69]:


# Use the best estimator to make predictions on test data

#Best parameters found:  {'activation': 'identity', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
#Best cross-validation score:  0.9174950484826297


best_mlp = grid_search.best_estimator_

# Predict on the test set
y_pred = best_mlp.predict(x_test2)


#Evaluation Report and Matrix
from sklearn.metrics import confusion_matrix, classification_report
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=target_names)
)


# # **Decision Tree**

# In[70]:


#Script for DecisionTree

from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=100)

# Train the classifier
dt_classifier.fit(x_train2, y_train)

# Make predictions
dt_predictions = dt_classifier.predict(x_test2)

# Evaluation Report and Matrix
print('\nDecision Tree Classifier:')
print('Confusion Matrix:')
print(confusion_matrix(y_test, dt_predictions))
print('\nClassification Report:')
print(classification_report(y_test, dt_predictions, target_names=target_names))

# Plot Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay.from_predictions(y_test, dt_predictions, display_labels=target_names, cmap=plt.cm.Blues)
disp.ax_.set_title("Decision Tree Confusion Matrix")
plt.show()


# ## **RANDOM FOREST**

# In[71]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=100)
rf.fit(x_train2, y_train)
predictions = rf.predict(x_test2)

# Evaluation Report and Matrix
print('\nRandom Forest Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('\nRandom Forest Classification Report:')
print(classification_report(y_test, predictions, target_names=target_names))


# In[73]:


#Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],  # Removed 'auto'
    'max_depth': [10, 20, 30, None],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=100)

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, n_jobs=-1, cv=3)

# Fit the model
grid_search_rf.fit(x_train2, y_train)

# Print the best parameters and score for Random Forest
print("Best parameters found for Random Forest: ", grid_search_rf.best_params_)
print("Best cross-validation score for Random Forest: ", grid_search_rf.best_score_)

# Use the best estimator to make predictions on test data
best_rf = grid_search_rf.best_estimator_

# Predict on the test set
y_pred_rf = best_rf.predict(x_test2)

# Evaluation Report and Matrix for Random Forest
print('\nConfusion Matrix for Random Forest:')
print(confusion_matrix(y_test, y_pred_rf))
print('\nClassification Report for Random Forest:')
print(classification_report(y_test, y_pred_rf, target_names=target_names, zero_division=0))


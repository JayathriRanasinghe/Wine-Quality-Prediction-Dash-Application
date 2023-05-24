#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# For data analysis
import pandas as pd
# For model creation and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# For visualizations and interactive dashboard creation
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[3]:


# Load dataset
data = pd.read_csv("winequality-red.csv")


# In[4]:


# check for missing values
print(data.isnull().sum())
# drop rows with missing values
data.dropna(inplace=True)


# In[5]:


# Drop duplicate rows
data.drop_duplicates(keep='first')


# In[6]:


# Check wine quality distribution
plt.figure(dpi=100)
sns.countplot(data=data, x="quality")
plt.xlabel("Count")
plt.ylabel("Quality Score")
plt.show()


# In[7]:


# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)


# In[8]:


# Calculate the correlation matrix
corr_matrix = data.corr()
# Plot heatmap
plt.figure(figsize=(12, 8), dpi=100)
sns.heatmap(corr_matrix, center=0, cmap='Blues', annot=True)
plt.show()


# In[9]:


# Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']


# In[10]:


# Split the data into training and testing sets (20% testing and 80% training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[11]:


# Create an object of the logistic regression model
logreg_model = LogisticRegression()


# In[12]:


# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# In[14]:


# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}

# Create a logistic regression model
logistic_model = LogisticRegression()

# Perform grid search
grid_search = GridSearchCV(logistic_model, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters
best_params = grid_search.best_params_

# Calculate F1 scores
initial_model = LogisticRegression()  # Instantiate a new logistic regression model
initial_model.fit(X_train, y_train)  # Fit the model on the training data
initial_model_f1 = f1_score(y_test, initial_model.predict(X_test))

best_model_f1 = f1_score(y_test, grid_search.best_estimator_.predict(X_test))

print("Initial Model F1 Score:", initial_model_f1)
print("Best Model F1 Score:", best_model_f1)

# Choose the best model
best_logistic_model = grid_search.best_estimator_
print(best_logistic_model)


# In[15]:


# Predict the labels of the test set
y_pred = logreg_model.predict(X_test)


# In[16]:


# Create the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)


# In[17]:


# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# Compute the precision of the model
precision = precision_score(y_test, y_pred)
# Compute the recall of the model
recall = recall_score(y_test, y_pred)
# Compute the F1 score of the model
f1 = f1_score(y_test, y_pred)
print(f1)


# In[19]:


# y_true and y_score are the true labels and predicted scores, respectively
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
plt.figure(dpi=100)
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[20]:

# Create the Dash app
app = dash.Dash(__name__)
# Define the layout of the dashboard
app.layout = html.Div(
children=[
html.H1('CO544-2023 Lab 3: Wine Quality Prediction',style={'backgroundColor':'blue'}),
# Layout for exploratory data analysis: correlation between two selected features
html.Div([
html.H3('Exploratory Data Analysis'),
html.Label('Feature 1 (X-axis)'),
dcc.Dropdown(
id='x_feature',
options=[{'label': col, 'value': col} for col in data.columns],
value=data.columns[0]
)
], style={'width': '30%', 'display': 'inline-block'}),
html.Div([
html.Label('Feature 2 (Y-axis)'),
dcc.Dropdown(
id='y_feature',
options=[{'label': col, 'value': col} for col in data.columns],
value=data.columns[1]
)
], style={'width': '30%', 'display': 'inline-block'}),
dcc.Graph(id='correlation_plot'),
# Layout for wine quality prediction based on input feature values
html.H3("Wine Quality Prediction"),
html.Div([
    html.Table([
        html.Tr([
            html.Td(html.Label("Fixed Acidity")),
            html.Td(dcc.Input(id='fixed_acidity', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Volatile Acidity")),
            html.Td(dcc.Input(id='volatile_acidity', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Citric Acid")),
            html.Td(dcc.Input(id='citric_acid', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Residual Sugar")),
            html.Td(dcc.Input(id='residual_sugar', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Chlorides")),
            html.Td(dcc.Input(id='chlorides', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Free Sulfur Dioxide")),
            html.Td(dcc.Input(id='free_sulfur_dioxide', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Total Sulfur Dioxide")),
            html.Td(dcc.Input(id='total_sulfur_dioxide', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Density")),
            html.Td(dcc.Input(id='density', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("pH")),
            html.Td(dcc.Input(id='ph', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Sulphates")),
            html.Td(dcc.Input(id='sulphates', type='number', required=True))
        ]),
        html.Tr([
            html.Td(html.Label("Alcohol")),
            html.Td(dcc.Input(id='alcohol', type='number', required=True))
        ])
    ])
])
,
html.Div([
html.Button('Predict', id='predict-button', n_clicks=0),
]),
html.Div([
html.H4("Predicted Quality"),
html.Div(id='prediction-output')
])
])



# In[21]:


# Define the callback to update the correlation plot
@app.callback(
  dash.dependencies.Output('correlation_plot', 'figure'),
  [dash.dependencies.Input('x_feature', 'value'),
  dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
  fig = px.scatter(data, x=x_feature, y=y_feature, color='quality')
  fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
  return fig
# Define the callback function to predict wine quality
@app.callback(
  Output(component_id='prediction-output', component_property='children'),
  [Input('predict-button', 'n_clicks')],
  [State('fixed_acidity', 'value'),
  State('volatile_acidity', 'value'),
  State('citric_acid', 'value'),
  State('residual_sugar', 'value'),
  State('chlorides', 'value'),
  State('free_sulfur_dioxide', 'value'),
  State('total_sulfur_dioxide', 'value'),
  State('density', 'value'),
  State('ph', 'value'),
  State('sulphates', 'value'),
  State('alcohol', 'value')]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid,
    residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, ph, sulphates, alcohol):
  # Create input features array for prediction
  input_features = np.array([fixed_acidity, volatile_acidity, citric_acid,
    residual_sugar, chlorides, free_sulfur_dioxide,
    total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1, -1)
  # Predict the wine quality (0 = bad, 1 = good)
  prediction = logreg_model.predict(input_features)[0]
  # Return the prediction
  if prediction == 1:
    return 'This wine is predicted to be good quality.'
  else:
    return 'This wine is predicted to be bad quality.'


# In[ ]:


if __name__ == '__main__':
  app.run_server(debug=False)


# In[ ]:





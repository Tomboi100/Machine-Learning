from random import randint

import knn as knn
import numpy as np
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.misc import face

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display the first few rows of the training data
train_df.head()
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Define features X and target y
X = train_df.drop('Class', axis=1)
y = train_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}
# # Create a random forest classifier
# rf = RandomForestClassifier()
# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(rf,
#                                  param_distributions = param_dist,
#                                  n_iter=5,
#                                  cv=5)
# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)

# Train the model
rfc.fit(X_train, y_train)

# Predict probabilities for the test set
y_prob = rfc.predict_proba(X_test)[:, 1]
y_test_prob = rfc.predict_proba(test_df.drop('TestId', axis=1))[:, 1]

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_prob)

print(roc_auc)
#print(y_prob)

# Create a DataFrame with the required structure
submission_df = pd.DataFrame({
    'TestId': test_df['TestId'],
    'PredictedScore': y_test_prob
})

# Save the DataFrame to a CSV file
submission_df.to_csv('datafilecsv.csv', index=False)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# gm = GaussianMixture(n_components=3, random_state=0).fit(X_train)
# #gm.predict(X)
# labels = gm.predict(X_train)
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.show()
#
# #transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 1.05253229]]
# X_aniso = np.dot(X_train, transformation)
# gm = GaussianMixture(n_components=3, random_state=0).fit(X_aniso)
# labels = gm.predict(X_train)
# plt.subplot(222)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=labels)
# plt.title("Anisotropicly Distributed Blobs")
# plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from geopy.distance import geodesic
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Loading dataset
dataset = pd.read_csv('FastagFraudDetection.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

# Extracting useful features from timestamp
dataset['Hour'] = dataset['Timestamp'].dt.hour
dataset['DayOfWeek'] = dataset['Timestamp'].dt.dayofweek
dataset['Month'] = dataset['Timestamp'].dt.month

# Dropping unnecessary columns
dataset = dataset.drop('Transaction_ID', axis=1)

# Applying One-hot encoding 
dataset = pd.get_dummies(dataset, columns=['Vehicle_Type', 'Lane_Type'])

# Label encoding
le = LabelEncoder()
dataset['Fraud_indicator'] = le.fit_transform(dataset['Fraud_indicator'])

# Feature Extraction Using Haversine Distance 
reference_point = (13.059816123454882, 77.77068662374292)
dataset['distance_from_city_center'] = dataset['Geographical_Location'].apply(
    lambda x: geodesic(reference_point, tuple(map(float, x.split(',')))).kilometers
)

# Scaling features
scaler = MinMaxScaler()
dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']] = scaler.fit_transform(
    dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']]
)

# Computing the correlation matrix
numeric_columns = dataset.select_dtypes(include=np.number).columns
correlation_matrix = dataset[numeric_columns].corr()

# Setting a correlation threshold of 0.03
correlation_threshold = 0.03
# Selecting features with absolute correlation above the threshold
selected_features = correlation_matrix[abs(correlation_matrix['Fraud_indicator']) > correlation_threshold].index

# Keeping only the selected features in the dataset
dataset = dataset[selected_features]

# Handling NaN values by filling with mean value
dataset.fillna(dataset.mean(), inplace=True)

# Plot a heatmap to visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

# Splitting data into features and labels
X = dataset.drop('Fraud_indicator', axis=1)
y = dataset['Fraud_indicator']

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print class distribution in the original dataset
print("Class Distribution in Original Dataset:")
print(y_train.value_counts())

# Oversampling using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Printing class distribution after oversampling
print("\nClass Distribution after Oversampling:")
print(y_train_resampled.value_counts())

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(estimator=BalancedRandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)
y_pred = best_model.predict(X_test)

# Extract GridSearchCV results
results = pd.DataFrame(grid_search.cv_results_)

# Plotting the results of the hyperparameter tuning
plt.figure(figsize=(12, 6))
sns.lineplot(data=results, x='param_n_estimators', y='mean_test_score', hue='param_max_depth', marker='o')
plt.title('Hyperparameter Tuning Results')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Test ROC AUC Score')
plt.legend(title='Max Depth')
plt.show()

# Evaluate the best model
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Saving the model
joblib.dump(best_model, 'model_FasttagFraudDetection.pkl')
joblib.dump(scaler, 'scaler.pkl')

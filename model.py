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

 
dataset = pd.read_csv('FastagFraudDetection.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
 
dataset['Hour'] = dataset['Timestamp'].dt.hour
dataset['DayOfWeek'] = dataset['Timestamp'].dt.dayofweek
dataset['Month'] = dataset['Timestamp'].dt.month

 
dataset = dataset.drop('Transaction_ID', axis=1)

 
dataset = pd.get_dummies(dataset, columns=['Vehicle_Type', 'Lane_Type'])


le = LabelEncoder()
dataset['Fraud_indicator'] = le.fit_transform(dataset['Fraud_indicator'])

reference_point = (13.059816123454882, 77.77068662374292)
dataset['distance_from_city_center'] = dataset['Geographical_Location'].apply(
    lambda x: geodesic(reference_point, tuple(map(float, x.split(',')))).kilometers
)

scaler = MinMaxScaler()
dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']] = scaler.fit_transform(
    dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']]
)

numeric_columns = dataset.select_dtypes(include=np.number).columns
correlation_matrix = dataset[numeric_columns].corr()

correlation_threshold = 0.03

selected_features = correlation_matrix[abs(correlation_matrix['Fraud_indicator']) > correlation_threshold].index


dataset = dataset[selected_features]

dataset.fillna(dataset.mean(), inplace=True)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

X = dataset.drop('Fraud_indicator', axis=1)
y = dataset['Fraud_indicator']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Class Distribution in Original Dataset:")
print(y_train.value_counts())

oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

print("\nClass Distribution after Oversampling:")
print(y_train_resampled.value_counts())

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

joblib.dump(best_model, 'model_FasttagFraudDetection.pkl')
joblib.dump(scaler, 'scaler.pkl')

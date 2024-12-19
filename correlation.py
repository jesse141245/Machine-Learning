import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            # Linear combination of inputs and weights
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid to get the predicted probabilities
            y_predicted = self.sigmoid(linear_model)

            # Calculate gradients for weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias using the learning rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Linear combination of inputs and weights
        linear_model = np.dot(X, self.weights) + self.bias
        # Apply sigmoid to get the predicted probabilities
        y_predicted_prob = self.sigmoid(linear_model)
        # Convert probabilities to binary outputs (0 or 1)
        y_predicted = [1 if i > 0.5 else 0 for i in y_predicted_prob]
        return np.array(y_predicted)
    def coef(self):
        # Return the weights (coefficients) and bias (intercept)
        return self.weights
    
scaler = StandardScaler()
data = pd.read_csv('output_rejected.csv')
interaction_list = []
interaction = pd.DataFrame()
length = data.shape[1]
for i in range(length - 2):
    for j in range(i + 1, length - 1):
        name = f"interaction*{data.columns[i]}*{data.columns[j]}"
        interaction_list.append(pd.DataFrame({name: data.iloc[:, i] * data.iloc[:, j]}))
interaction = pd.concat(interaction_list, axis=1)
original_column_order = list(interaction.columns)
print(interaction.head())
y = data['All rejection']
print(y)
model = LogisticRegressionScratch()
interaction = scaler.fit_transform(interaction)
model.fit(interaction, y)
importance = np.abs(model.coef().reshape(-1))
print(len(importance))
importance_df = pd.DataFrame({
    'Feature': original_column_order,
    'Importance': importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df.head(20))
output = pd.read_csv("datasets/selected_features_orig.csv")
split_features = importance_df.head(20)['Feature'].str.split('*', n=2, expand=True)
features = split_features[1].drop_duplicates().tolist() + split_features[2].drop_duplicates().tolist()
print(features)
filtered_data = data[features]
new_columns = [col for col in filtered_data.columns if col not in output.columns]
if new_columns:
    filtered_data_new = filtered_data[new_columns]
    merged_data = pd.concat([filtered_data_new, output], axis=1)
    merged_data = pd.concat([merged_data, data['All rejection']], axis=1)

else:
    merged_data = output
    merged_data = pd.concat([merged_data, data['All rejection']], axis=1) 
merged_data = merged_data.drop_duplicates()
merged_data.to_csv("datasets/selected_features_orig.csv", index=False)

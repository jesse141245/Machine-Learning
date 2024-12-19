import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset (assuming your dataset loading method is correct)
data = pd.read_csv('datasets/rejected.csv')

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Label encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Standard scale the numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Combine the scaled numerical features and encoded categorical features
X_transformed_df = X.copy()

# Add the target column back to the DataFrame
X_transformed_df['All rejection'] = y.values

# Save the preprocessed data to a new CSV file
X_transformed_df.to_csv('output_rejected.csv', index=False)

# Print the preprocessed data
print("Preprocessed Data:")
print(X_transformed_df.head())

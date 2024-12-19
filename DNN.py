# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn module for SMOTE
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Category Encoders
from category_encoders import TargetEncoder

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Load your dataset
data = pd.read_csv('datasets/mayo_f_lowres_CJ.csv')  # Replace with your dataset file

# 2. Separate features and target
X = data.drop('outcome', axis=1)  # Replace 'outcome' with your target column
y = data['outcome']

# 3. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 4. Handle 'Date_of_Transplant' separately
if 'Date_of_Transplant' in categorical_features:
    categorical_features.remove('Date_of_Transplant')
    # Convert 'Date_of_Transplant' to datetime
    X['Date_of_Transplant'] = pd.to_datetime(X['Date_of_Transplant'], errors='coerce')
    # Extract date features
    X['Transplant_Year'] = X['Date_of_Transplant'].dt.year
    X['Transplant_Month'] = X['Date_of_Transplant'].dt.month
    X['Transplant_Day'] = X['Date_of_Transplant'].dt.day
    # Drop the original 'Date_of_Transplant' column
    X = X.drop('Date_of_Transplant', axis=1)
    # Add new features to numeric_features
    numeric_features.extend(['Transplant_Year', 'Transplant_Month', 'Transplant_Day'])

# 5. Preprocess numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),    # Handle missing values
    ('scaler', StandardScaler())                    # Standardize numeric features
])

# 6. Preprocess categorical features
# Impute missing values
X_categorical = X[categorical_features].copy()
imputer = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(imputer.fit_transform(X_categorical), columns=categorical_features)

# Handle high-cardinality categorical features using Target Encoding
# Identify high-cardinality categorical features
high_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() > 10]
low_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() <= 10]

# Target Encoding for high-cardinality features
X_high_cardinality = X_categorical[high_cardinality_cols]
te = TargetEncoder(cols=high_cardinality_cols)
X_high_cardinality_encoded = te.fit_transform(X_high_cardinality, y)

# One-hot encoding for low-cardinality features
X_low_cardinality = X_categorical[low_cardinality_cols]
X_low_cardinality_encoded = pd.get_dummies(X_low_cardinality, drop_first=True)

# Combine all preprocessed features
X_preprocessed_numeric = numeric_transformer.fit_transform(X[numeric_features])
X_preprocessed = np.hstack([
    X_preprocessed_numeric,
    X_high_cardinality_encoded.values,
    X_low_cardinality_encoded.values
])

# Update feature names
numeric_feature_names = numeric_features
high_cardinality_feature_names = high_cardinality_cols
low_cardinality_feature_names = X_low_cardinality_encoded.columns.tolist()
feature_names = numeric_feature_names + high_cardinality_feature_names + low_cardinality_feature_names

# 7. Split into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# 8. Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("Resampled dataset shape:", X_resampled.shape, y_resampled.shape)

# Hyperparameter tuning ranges for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0]
}

# Hyperparameter tuning ranges for the DNN
dnn_param_grid = {
    'learning_rate': [0.0001, 0.0005],
    'l2_regularizer': [0.0001, 0.001],
    'num_neurons': [32, 64],
    'dropout_rate': [0.3, 0.5],
    'batch_size': [32, 64]
}

# To store the best models and performance
best_auc = 0
best_params = {}
best_model = None

# Loop over XGBoost hyperparameters
import itertools

xgb_param_combinations = list(itertools.product(
    xgb_param_grid['n_estimators'],
    xgb_param_grid['max_depth'],
    xgb_param_grid['learning_rate'],
    xgb_param_grid['subsample']
))

for n_estimators, max_depth, learning_rate, subsample in xgb_param_combinations:
    print(f"\nTraining XGBoost with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}")
    # 9. Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_resampled, y_resampled)

    # 10. Extract leaf indices as features
    # For training data
    X_train_leaf = xgb_model.apply(X_resampled)
    X_train_leaf = X_train_leaf.reshape(-1, xgb_model.n_estimators)

    # For test data
    X_test_leaf = xgb_model.apply(X_test)
    X_test_leaf = X_test_leaf.reshape(-1, xgb_model.n_estimators)

    # 11. One-hot encode the leaf indices
    # Initialize OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit on training leaf indices
    X_train_leaf_ohe = ohe.fit_transform(X_train_leaf)
    X_test_leaf_ohe = ohe.transform(X_test_leaf)

    # 12. Combine original features with XGBoost leaf features
    X_train_combined = np.hstack([X_resampled, X_train_leaf_ohe])
    X_test_combined = np.hstack([X_test, X_test_leaf_ohe])

    print("Combined training set shape:", X_train_combined.shape)
    print("Combined test set shape:", X_test_combined.shape)

    # Loop over DNN hyperparameters
    dnn_param_combinations = list(itertools.product(
        dnn_param_grid['learning_rate'],
        dnn_param_grid['l2_regularizer'],
        dnn_param_grid['num_neurons'],
        dnn_param_grid['dropout_rate'],
        dnn_param_grid['batch_size']
    ))

    for lr, l2_reg, num_neurons, dropout_rate, batch_size in dnn_param_combinations:
        print(f"Training DNN with learning_rate={lr}, l2_regularizer={l2_reg}, num_neurons={num_neurons}, dropout_rate={dropout_rate}, batch_size={batch_size}")
        # 13. Define the DNN model with adjusted parameters
        from tensorflow.keras.regularizers import l2

        input_dim = X_train_combined.shape[1]

        model = Sequential([
            Dense(num_neurons, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(num_neurons // 2, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])

        # 14. Compile the model with adjusted learning rate and focal loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        def focal_loss(alpha=0.25, gamma=2.0):
            def focal_loss_fixed(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
                y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
                pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                loss = alpha * tf.pow(1 - pt, gamma) * bce
                return tf.reduce_mean(loss)
            return focal_loss_fixed

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=focal_loss(alpha=0.25, gamma=2.0),
                      metrics=['accuracy'])

        # 15. Add early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 16. Train the model without class weights
        history = model.fit(X_train_combined, y_resampled,
                            validation_split=0.2,
                            epochs=50,
                            batch_size=batch_size,
                            callbacks=[early_stopping],
                            verbose=0)

        # 17. Evaluate the model
        y_pred_prob = model.predict(X_test_combined).ravel()

        # Calculate AUC-ROC
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Update best model if current model has better AUC
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_params = {
                'xgb_n_estimators': n_estimators,
                'xgb_max_depth': max_depth,
                'xgb_learning_rate': learning_rate,
                'xgb_subsample': subsample,
                'dnn_learning_rate': lr,
                'dnn_l2_regularizer': l2_reg,
                'dnn_num_neurons': num_neurons,
                'dnn_dropout_rate': dropout_rate,
                'dnn_batch_size': batch_size
            }
            best_model = model
            # Save predictions and actual labels
            best_y_pred_prob = y_pred_prob
            best_X_test_combined = X_test_combined

        print(f"Validation AUC-ROC: {roc_auc:.4f}")

# After hyperparameter tuning
print("\nBest Model Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Evaluate the best model
print("\nEvaluating Best Model on Test Data...")
y_pred_prob = best_model.predict(best_X_test_combined).ravel()

# Adjust threshold using Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Use the optimal threshold to make final predictions
y_pred = (y_pred_prob >= optimal_threshold).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# AUC-ROC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {roc_auc:.4f}")

# AUC-PR Score
auc_pr = average_precision_score(y_test, y_pred_prob)
print(f"AUC-PR: {auc_pr:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (AUC = {auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the best model
best_model.save('best_xgboost_dnn_combined_model.h5')
print("Best Combined XGBoost and DNN model saved as 'best_xgboost_dnn_combined_model.h5'.")

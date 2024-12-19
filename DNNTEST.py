# Install if needed:
# !pip install optuna imblearn category_encoders xgboost focal-loss

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn modules
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Imbalanced-learn module
from imblearn.combine import SMOTETomek

# XGBoost
import xgboost as xgb

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Category Encoders
from category_encoders import TargetEncoder

# Optuna for hyperparameter tuning
import optuna

#---------------------------------------------
# Load your dataset
data = pd.read_csv('datasets/mayo_f_lowres_CJ.csv')  # Adjust to your file
X = data.drop('outcome', axis=1)
y = data['outcome']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle 'Date_of_Transplant' if present
if 'Date_of_Transplant' in categorical_features:
    categorical_features.remove('Date_of_Transplant')
    X['Date_of_Transplant'] = pd.to_datetime(X['Date_of_Transplant'], errors='coerce')
    X['Transplant_Year'] = X['Date_of_Transplant'].dt.year
    X['Transplant_Month'] = X['Date_of_Transplant'].dt.month
    X['Transplant_Day'] = X['Date_of_Transplant'].dt.day
    X = X.drop('Date_of_Transplant', axis=1)
    numeric_features.extend(['Transplant_Year', 'Transplant_Month', 'Transplant_Day'])

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocess categorical features
X_categorical = X[categorical_features].copy()
imputer = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(imputer.fit_transform(X_categorical), columns=categorical_features)

# High-cardinality vs low-cardinality
high_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() > 10]
low_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() <= 10]

# Target Encoding for high-cardinality features
te = TargetEncoder(cols=high_cardinality_cols)
X_high_cardinality_encoded = te.fit_transform(X_categorical[high_cardinality_cols], y)

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

numeric_feature_names = numeric_features
high_cardinality_feature_names = high_cardinality_cols
low_cardinality_feature_names = X_low_cardinality_encoded.columns.tolist()
feature_names = numeric_feature_names + high_cardinality_feature_names + low_cardinality_feature_names

#------------------------------------------------------
# We will define an objective function for Optuna
# This function will:
# 1. Sample hyperparams for XGBoost
# 2. Train XGBoost, get leaf embeddings
# 3. Sample hyperparams for DNN + focal loss
# 4. Train DNN using stratified k-fold CV to get mean AUC
# 5. Return mean AUC to Optuna

def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow((1 - pt), gamma) * bce
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def create_dnn(input_dim, num_neurons, dropout_rate, l2_reg, alpha, gamma, lr):
    model = Sequential([
        Dense(num_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_neurons // 2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss=focal_loss(alpha=alpha, gamma=gamma),
                  metrics=['accuracy'])
    return model

def objective(trial):
    # Hyperparameters for XGBoost
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 100, 600, step=100)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.001, 0.1, log=True)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.01, 10.0, log=True)

    # Hyperparameters for DNN
    dnn_lr = trial.suggest_float("dnn_lr", 1e-4, 5e-3, log=True)
    dnn_l2_regularizer = trial.suggest_float("dnn_l2_regularizer", 1e-5, 1e-2, log=True)
    dnn_num_neurons = trial.suggest_int("dnn_num_neurons", 32, 256, step=32)
    dnn_dropout_rate = trial.suggest_float("dnn_dropout_rate", 0.1, 0.5, step=0.1)
    alpha = trial.suggest_float("focal_alpha", 0.1, 0.5, step=0.1)
    gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)
    batch_size = trial.suggest_categorical("dnn_batch_size", [32, 64, 128])

    # Data balancing method
    smt = SMOTETomek(random_state=42)

    # Perform Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, val_index in skf.split(X_preprocessed, y):
        X_train_fold, X_val_fold = X_preprocessed[train_index], X_preprocessed[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Resample training data
        X_res, y_res = smt.fit_resample(X_train_fold, y_train_fold)

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            reg_lambda=xgb_reg_lambda,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_res, y_res)

        # Extract leaf embeddings
        X_train_leaf = xgb_model.apply(X_res).reshape(-1, xgb_model.n_estimators)
        X_val_leaf = xgb_model.apply(X_val_fold).reshape(-1, xgb_model.n_estimators)

        # One-hot encode leaf indices
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_leaf_ohe = ohe.fit_transform(X_train_leaf)
        X_val_leaf_ohe = ohe.transform(X_val_leaf)

        # Combine original features with leaf features
        X_train_combined = np.hstack([X_res, X_train_leaf_ohe])
        X_val_combined = np.hstack([X_val_fold, X_val_leaf_ohe])

        # Build DNN
        input_dim = X_train_combined.shape[1]
        model = create_dnn(input_dim, dnn_num_neurons, dnn_dropout_rate, dnn_l2_regularizer, alpha, gamma, dnn_lr)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train_combined, y_res,
                  validation_split=0.2,
                  epochs=50,
                  batch_size=batch_size,
                  callbacks=[early_stopping],
                  verbose=0)

        y_pred_prob = model.predict(X_val_combined).ravel()
        fold_auc = roc_auc_score(y_val_fold, y_pred_prob)
        auc_scores.append(fold_auc)

    mean_auc = np.mean(auc_scores)
    return mean_auc

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2000, show_progress_bar=True)

print("Best Trial:")
trial = study.best_trial
print(trial.values)
print(trial.params)

# After finding the best hyperparameters, retrain on the full training set and evaluate on a hold-out set (if you have one).
# Here we just retrain on the entire dataset using the best parameters for demonstration.

best_params = study.best_params

# Split the data into final train/test if you have a hold-out set
# In this example, we will just do a final train-test split.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

xgb_model_final = xgb.XGBClassifier(
    n_estimators=best_params['xgb_n_estimators'],
    max_depth=best_params['xgb_max_depth'],
    learning_rate=best_params['xgb_learning_rate'],
    subsample=best_params['xgb_subsample'],
    reg_lambda=best_params['xgb_reg_lambda'],
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model_final.fit(X_resampled, y_resampled)

X_train_leaf = xgb_model_final.apply(X_resampled).reshape(-1, xgb_model_final.n_estimators)
X_test_leaf = xgb_model_final.apply(X_test).reshape(-1, xgb_model_final.n_estimators)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_leaf_ohe = ohe.fit_transform(X_train_leaf)
X_test_leaf_ohe = ohe.transform(X_test_leaf)

X_train_combined = np.hstack([X_resampled, X_train_leaf_ohe])
X_test_combined = np.hstack([X_test, X_test_leaf_ohe])

model_final = create_dnn(
    input_dim=X_train_combined.shape[1],
    num_neurons=best_params['dnn_num_neurons'],
    dropout_rate=best_params['dnn_dropout_rate'],
    l2_reg=best_params['dnn_l2_regularizer'],
    alpha=best_params['focal_alpha'],
    gamma=best_params['focal_gamma'],
    lr=best_params['dnn_lr']
)

early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_final.fit(X_train_combined, y_resampled, validation_split=0.2, epochs=50, batch_size=best_params['dnn_batch_size'], callbacks=[early_stopping_final], verbose=0)

y_pred_prob = model_final.predict(X_test_combined).ravel()
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"Final Test AUC-ROC: {roc_auc:.4f}")

# Find the optimal threshold based on PR curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
y_pred = (y_pred_prob >= optimal_threshold).astype(int)

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
auc_pr = average_precision_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (AUC = {auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the final model
model_final.save('best_xgboost_dnn_combined_model.h5')
print("Best Combined XGBoost and DNN model saved as 'best_xgboost_dnn_combined_model.h5'.")

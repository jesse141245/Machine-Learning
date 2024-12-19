
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve
)
from sklearn.pipeline import Pipeline

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import optuna

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from category_encoders import TargetEncoder

data = pd.read_csv('datasets/mayo_az_highres_ml_final_h1h2.csv')
X = data.drop('outcome', axis=1)
y = data['outcome']

le = LabelEncoder()
y = le.fit_transform(y)  
num_classes = len(np.unique(y))

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

if 'Date_of_Transplant' in categorical_features:
    categorical_features.remove('Date_of_Transplant')
    X['Date_of_Transplant'] = pd.to_datetime(X['Date_of_Transplant'], errors='coerce')
    X['Transplant_Year'] = X['Date_of_Transplant'].dt.year
    X['Transplant_Month'] = X['Date_of_Transplant'].dt.month
    X['Transplant_Day'] = X['Date_of_Transplant'].dt.day
    X = X.drop('Date_of_Transplant', axis=1)
    numeric_features.extend(['Transplant_Year', 'Transplant_Month', 'Transplant_Day'])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_categorical = X[categorical_features].copy()
imputer = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(imputer.fit_transform(X_categorical), columns=categorical_features)

high_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() > 10]
low_cardinality_cols = [col for col in categorical_features if X_categorical[col].nunique() <= 10]

te = TargetEncoder(cols=high_cardinality_cols)
X_high_cardinality_encoded = te.fit_transform(X_categorical[high_cardinality_cols], y)

X_low_cardinality = X_categorical[low_cardinality_cols]
X_low_cardinality_encoded = pd.get_dummies(X_low_cardinality, drop_first=True)

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

X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, stratify=y)

def create_dnn(input_dim, num_neurons, dropout_rate, l2_reg, lr):
    model = Sequential([
        Dense(num_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_neurons // 2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def flatten_leaf_indices(leaf_indices):
    if leaf_indices.ndim == 3:
        n_samples, n_estimators, n_outputs = leaf_indices.shape
        leaf_indices = leaf_indices.reshape(n_samples, n_estimators * n_outputs)
    return leaf_indices

def objective(trial):
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 100, 500, step=100)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.001, 0.3, log=True)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.01, 10.0, log=True)

    dnn_lr = trial.suggest_float("dnn_lr", 1e-5, 1e-2, log=True)
    dnn_l2_regularizer = trial.suggest_float("dnn_l2_regularizer", 1e-6, 1e-2, log=True)
    dnn_num_neurons = trial.suggest_int("dnn_num_neurons", 32, 256, step=32)
    dnn_dropout_rate = trial.suggest_float("dnn_dropout_rate", 0.1, 0.6, step=0.1)
    batch_size = trial.suggest_categorical("dnn_batch_size", [32, 64, 128])

    smote = SMOTE(k_neighbors=2)
    smt = SMOTETomek(smote=smote)
    X_res, y_res = smt.fit_resample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        reg_lambda=xgb_reg_lambda,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    xgb_model.fit(X_res, y_res)

    X_train_leaf = xgb_model.apply(X_res)
    X_train_leaf = flatten_leaf_indices(X_train_leaf)

    X_val_leaf = xgb_model.apply(X_val)
    X_val_leaf = flatten_leaf_indices(X_val_leaf)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_leaf_ohe = ohe.fit_transform(X_train_leaf)
    X_val_leaf_ohe = ohe.transform(X_val_leaf)

    X_train_combined = np.hstack([X_res, X_train_leaf_ohe])
    X_val_combined = np.hstack([X_val, X_val_leaf_ohe])

    input_dim = X_train_combined.shape[1]
    model = create_dnn(input_dim, dnn_num_neurons, dnn_dropout_rate, dnn_l2_regularizer, dnn_lr)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train_combined, y_res,
              validation_data=(X_val_combined, y_val),
              epochs=50,
              batch_size=batch_size,
              callbacks=[early_stopping],
              verbose=0)

    y_pred_prob = model.predict(X_val_combined)
    val_auc = roc_auc_score((y_val == 1).astype(int), y_pred_prob[:, 0])
    fpr, tpr, _ = roc_curve((y_val == 1).astype(int), y_pred_prob[:, 0])
    class0_auc = roc_auc_score((y_val == 1).astype(int), y_pred_prob[:, 0])

    y_pred = np.argmax(y_pred_prob, axis=1)
    cm = confusion_matrix(y_val, y_pred)

    trial.set_user_attr("fpr", fpr)
    trial.set_user_attr("tpr", tpr)
    trial.set_user_attr("class0_auc", class0_auc)
    trial.set_user_attr("cm", cm)

    return val_auc

plt.ion()

trial_values = []

def callback_for_improvement(study, trial):
    if trial.value is not None:
        trial_values.append(trial.value)

    if study.best_trial.number == trial.number:
        fpr = trial.user_attrs["fpr"]
        tpr = trial.user_attrs["tpr"]
        class0_auc = trial.user_attrs["class0_auc"]
        cm = trial.user_attrs["cm"]  

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Class 0 ROC Curve (AUC = {class0_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Class 0 vs Rest)')
        plt.legend()

        filename_roc = f"best_trial_{0}_roc_curve.png"
        plt.savefig(filename_roc)
        plt.close()
        print(f"New best trial: {trial.number} with AUC={trial.value:.4f}. ROC curve saved to {filename_roc}.")

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix (Validation Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        filename_cm = f"best_trial_{0}_confusion_matrix.png"
        plt.savefig(filename_cm)
        plt.close()
        print(f"Confusion matrix saved to {filename_cm}.")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, callbacks=[callback_for_improvement], show_progress_bar=True)

print("Best Trial:")
trial = study.best_trial
print("Best Trial Value (AUC):", trial.value)
print("Best Trial Params:", trial.params)

df = study.trials_dataframe()
df.to_csv("trials_data")
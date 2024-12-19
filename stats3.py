import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertTokenizer,
    BertConfig,
    BertPreTrainedModel,
    BertModel,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Define the custom model class
class BertWithNumericalFeatures(BertPreTrainedModel):
    def __init__(self, config, class_weights=None):
        super(BertWithNumericalFeatures, self).__init__(config)
        self.num_numerical_features = config.num_numerical_features
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.num_labels = config.num_labels
        self.class_weights = class_weights

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.3)

        # Increase model complexity
        input_size = config.hidden_size + self.num_numerical_features
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, self.num_labels)

        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU(alpha=1.0)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        numerical_features=None,
        labels=None,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if numerical_features is not None and self.num_numerical_features > 0:
            numerical_features = numerical_features.to(cls_output.device)
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
        else:
            combined_output = cls_output

        combined_output = self.dropout(combined_output)

        # Pass through the additional fully connected layers
        x = self.fc1(combined_output)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc5(x)

        return logits

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_len):
        self.texts = texts
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.numerical_features is not None:
            numerical = self.numerical_features[idx]
            numerical = torch.tensor(numerical, dtype=torch.float)
        else:
            numerical = None

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "numerical_features": numerical,
            "labels": torch.tensor(label, dtype=torch.long),
        }

def main():
    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2 or 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    data_path = 'validation_data_fine_tune951.csv'
    model_save_path = 'Mayo_Model951'

    # Load configuration and tokenizer
    config = BertConfig.from_pretrained(model_save_path)
    df_test = pd.read_csv(data_path)
    print(f"Loaded test data from {data_path} with shape {df_test.shape}")

    # Identify numerical and text columns
    numerical_columns = df_test.select_dtypes(include=["float64", "int64"]).columns.tolist()
    text_columns = df_test.select_dtypes(include=["object"]).columns
    num_numerical_features = len(numerical_columns)
    # Add custom attributes
    config.num_labels = 2  # Set to the correct number of labels
    config.alpha = 0.75  # Add alpha if not present
    config.gamma = 2.0  # Add gamma if not present
    tokenizer = BertTokenizer.from_pretrained(model_save_path)

    print(f"Numerical columns ({num_numerical_features}): {numerical_columns}")
    print(f"Text columns: {list(text_columns)}")

    # Prepare data (texts, numerical features, labels)
    texts = df_test[text_columns].apply(
        lambda x: " ".join(x.astype(str)), axis=1
    ).tolist()

    if num_numerical_features > 0:
        numerical_features = df_test[numerical_columns].values

        # Handle missing values (if any)
        numerical_features = np.nan_to_num(numerical_features)  # Replace NaNs with 0

        # Load scaler and PCA
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('pca_transform.pkl', 'rb') as f:
            pca = pickle.load(f)

        # Transform features
        numerical_features = scaler.transform(numerical_features)
        numerical_features = pca.transform(numerical_features)

        # Update num_numerical_features after PCA
        num_numerical_features = numerical_features.shape[1]
    else:
        print("No numerical features found, creating an empty array.")
        numerical_features = np.empty((len(df_test), 0))
        num_numerical_features = 0

    # Update config with the new number of numerical features
    config.num_numerical_features = num_numerical_features

    # Initialize the model with the updated config
    model = BertWithNumericalFeatures.from_pretrained(
        model_save_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()
    print("Loaded fine-tuned model.")

    labels = df_test.iloc[:, -1].tolist()

    # Create dataset and dataloader
    val_dataset = CustomDataset(
        texts,
        numerical_features,
        labels,
        tokenizer,
        MAX_LEN,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            numerical_features_batch = (
                batch["numerical_features"].to(device)
                if batch["numerical_features"] is not None
                else None
            )

            # Debugging: Print device information
            # print(f"Input IDs device: {input_ids.device}")
            # print(f"Attention mask device: {attention_mask.device}")
            # if numerical_features_batch is not None:
            #     print(f"Numerical features device: {numerical_features_batch.device}")
            # print(f"Model device: {next(model.parameters()).device}")

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features_batch,
            )
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Assuming class 1 is positive
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())

    # Compute and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, fignum=1)
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Classification report
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    background = next(iter(val_loader))
    background_input_ids = background['input_ids'][:100].to(device)
    background_attention_mask = background['attention_mask'][:100].to(device)
    background_numerical_features = (
        background["numerical_features"][:100].to(device)
        if background["numerical_features"] is not None
        else None
    )
    explainer = shap.DeepExplainer(model, [background_input_ids, background_attention_mask, background_numerical_features])
    shap_values = explainer.shap_values([batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['numerical_features'].to(device)])
    shap.summary_plot(shap_values, features=[batch['input_ids'], 'numerical_features'], feature_names=['input_ids', 'numerical_features'])

if __name__ == '__main__':
    main()

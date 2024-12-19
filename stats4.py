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

        input_size = config.hidden_size + self.num_numerical_features
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, self.num_labels)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU(alpha=1.0)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        numerical_features=None,
        labels=None,
    ):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if numerical_features is not None and self.num_numerical_features > 0:
            numerical_features = numerical_features.to(cls_output.device)
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
        else:
            combined_output = cls_output

        combined_output = self.dropout(combined_output)

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
    data_path = 'validation_data_fine_tunePC1.csv'
    model_save_path = 'Fine_Tuned_ModelPC1'

    config = BertConfig.from_pretrained(model_save_path)
    df_test = pd.read_csv(data_path)
    print(f"Loaded test data from {data_path} with shape {df_test.shape}")

    # Identify numerical and text columns
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    numerical_features = df_test[feature_names].values
    numerical_features = np.nan_to_num(numerical_features)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca_transform.pkl', 'rb') as f:
        pca = pickle.load(f)

    numerical_features = scaler.transform(numerical_features)
    numerical_features = pca.transform(numerical_features)
    num_numerical_features = numerical_features.shape[1]

    config.num_numerical_features = num_numerical_features
    config.num_labels = 2
    config.alpha = 0.75
    config.gamma = 2.0

    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    model = BertWithNumericalFeatures.from_pretrained(
        model_save_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()

    texts = df_test.apply(lambda x: " ".join(x.astype(str)), axis=1).tolist()
    labels = df_test.iloc[:, -1].tolist()

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

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            numerical_features_batch = batch["numerical_features"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features_batch,
            )
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    plt.figure(figsize=(6, 6))
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, fignum=1)
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(classification_report(all_labels, all_preds, zero_division=0))

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


if __name__ == '__main__':
    main()

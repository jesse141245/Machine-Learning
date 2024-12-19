import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertConfig,
    BertPreTrainedModel,
    BertModel,

)
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Custom model class
class BertWithNumericalFeatures(BertPreTrainedModel):
    def __init__(self, config, class_weights=None, **kwargs):
        super(BertWithNumericalFeatures, self).__init__(config)
        self.num_numerical_features = config.num_numerical_features
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.num_labels = config.num_labels

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

        if labels is not None:
            # Loss computation can be added if needed
            return logits
        else:
            return logits


class CustomDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_len):
        self.texts = texts
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Standardize numerical features if they exist
        if self.numerical_features.shape[1] > 0:
            self.scaler = StandardScaler()
            self.numerical_features = self.scaler.fit_transform(
                self.numerical_features
            )
        else:
            self.numerical_features = None

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
    # Configurations
    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2 or 1

    # Paths
    fine_tuned_model_path = "Mayo_ModelPC1"
    test_data_path = "validation_data_fine_tune.csv"

    # Load the fine-tuned model and tokenizer
    config = BertConfig.from_pretrained(fine_tuned_model_path)
    tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)

    # Load the model
    model = BertWithNumericalFeatures.from_pretrained(
        fine_tuned_model_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()
    print("Loaded fine-tuned model.")

    # Load test data
    df_test = pd.read_csv(test_data_path)
    print(f"Loaded test data from {test_data_path} with shape {df_test.shape}")

    # Identify numerical and text columns
    numerical_columns = df_test.select_dtypes(include=["float64", "int64"]).columns.tolist()
    text_columns = df_test.select_dtypes(include=["object"]).columns

    num_numerical_features = len(numerical_columns)
    print(f"Numerical columns ({num_numerical_features}): {numerical_columns}")
    print(f"Text columns: {list(text_columns)}")

    # Prepare data (texts, numerical features, labels)
    texts = df_test[text_columns].apply(
        lambda x: " ".join(x.astype(str)), axis=1
    ).tolist()

    if num_numerical_features > 0:
        numerical_features = df_test[numerical_columns].values
    else:
        numerical_features = np.empty((len(df_test), 0))

    labels = df_test.iloc[:, -1].tolist()  # Assuming label is the last column

    # Create dataset and dataloader
    test_dataset = CustomDataset(
        texts,
        numerical_features,
        labels,
        tokenizer,
        MAX_LEN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize lists to store labels and predictions
    all_labels = []
    all_preds = []
    all_probs = []

    # Disable gradient computation
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            numerical_features = (
                batch["numerical_features"].to(device)
                if batch["numerical_features"] is not None
                else None
            )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
            )

            probs = torch.softmax(logits, dim=1)
            class_1_probs = probs[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(class_1_probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Optional: Save ROC curve plot
    # plt.savefig('roc_curve.png')

    # Plot Confusion Matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(all_labels)))
    plt.xticks(tick_marks, np.unique(all_labels))
    plt.yticks(tick_marks, np.unique(all_labels))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    # Optional: Save Confusion Matrix plot
    # plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    main()

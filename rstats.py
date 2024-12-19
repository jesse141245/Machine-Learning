import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shap  # SHAP library
from sklearn.utils.class_weight import compute_class_weight

# Define the custom model class (same as your training script)
class RobertaWithNumericalFeatures(nn.Module):
    def __init__(self, roberta_model_name, num_labels, num_numerical_features, class_weights, alpha=0.75, gamma=2.0):
        super(RobertaWithNumericalFeatures, self).__init__()
        self.num_numerical_features = num_numerical_features
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(p=0.3)
        
        # Fully connected layers for combined features
        self.fc1 = nn.Linear(self.roberta.config.hidden_size + num_numerical_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_labels)
                
        # Activation function
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Initialize WeightedFocalLoss with class_weights
        self.loss_fn = WeightedFocalLoss(class_weights=class_weights, alpha=alpha, gamma=gamma)

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = roberta_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation

        if numerical_features is not None:
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
            combined_output = self.dropout(combined_output)
        else:
            combined_output = cls_output

        # Pass through the additional fully connected layers
        x = self.fc1(combined_output)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc3(x)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
    
class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5, gamma=1.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).float()
        pt = torch.sum(probs * labels_one_hot, dim=1)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        return loss.mean()
# Define the custom dataset class (same as your training script)
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
        numerical = self.numerical_features[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(numerical, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    data_path = 'datasets/val_data.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()
    labels_array = np.array(result)
    model_save_path = 'Roberta_Model'

    tokenizer = RobertaTokenizer.from_pretrained(model_save_path)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")
    model = RobertaWithNumericalFeatures(
            roberta_model_name='roberta-base', 
            num_labels=2, 
            num_numerical_features=len(numerical_columns),
            class_weights=class_weights,
            alpha=0.75,
            gamma=2.0
    )

    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2


    val_dataset = CustomDataset(texts, numerical_features, result, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask)
            
            # Make sure you extract the logits correctly
            logits = outputs if not isinstance(outputs, tuple) else outputs[1]

            # Calculate probabilities
            probs = torch.softmax(logits, dim=-1)

            # Ensure `probs` has the right shape to index the positive class probabilities
            if len(probs.shape) == 2 and probs.shape[1] > 1:
                all_probs.extend(probs[:, 1].cpu().numpy())  # Assuming class 1 is positive
            elif len(probs.shape) == 1:
                all_probs.extend(probs.cpu().numpy())  # If only one class output
            else:
                raise ValueError(f"Unexpected probs shape: {probs.shape}")

            # Add labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())

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
    print(classification_report(all_labels, all_preds))

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

    # SHAP values for interpretability
    explainer = shap.DeepExplainer(model, torch.cat([batch['input_ids'].to(device) for batch in val_loader]))
    shap_values = explainer.shap_values(torch.cat([batch['input_ids'].to(device) for batch in val_loader]))

    shap.summary_plot(shap_values, torch.cat([batch['input_ids'] for batch in val_loader], dim=0).cpu())

if __name__ == '__main__':
    main()

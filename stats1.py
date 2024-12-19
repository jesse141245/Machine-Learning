import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
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
class BertWithNumericalFeatures(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_numerical_features, class_weights, alpha=0.75, gamma=1.5):
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        super(BertWithNumericalFeatures, self).__init__(config=bert_model.config)
        self.num_numerical_features = num_numerical_features
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate to 0.5 for better regularization
        
        # Simplify model architecture to reduce complexity
        self.fc1 = nn.Linear(self.config.hidden_size + num_numerical_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_labels)
                
        # Activation function
        self.relu = nn.ReLU()

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Initialize Hybrid Loss with CrossEntropyLoss and Focal Loss
        self.ce_loss_fn = CrossEntropyLoss()
        self.focal_loss_fn = WeightedFocalLoss(class_weights=class_weights, alpha=alpha, gamma=gamma)
        
        # Pre-trained BERT model
        self.bert = bert_model.bert

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None, temperature=1.0):
        # BERT model forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[0][:, 0, :]  # Use [CLS] token representation

        # Concatenate CLS output with numerical features if available
        if numerical_features is not None:
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
            combined_output = self.dropout(combined_output)
        else:
            combined_output = cls_output

        # Pass through the additional fully connected layers
        x = self.fc1(combined_output)
        x = self.bn1(x)             # Batch Normalization after the first FC layer
        x = self.relu(x)            # Activation Function (ReLU)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)             # Batch Normalization after the second FC layer
        x = self.relu(x)            # Activation Function (ReLU)
        x = self.dropout(x)

        # Output layer
        logits = self.fc3(x)
        logits = logits / temperature  # Apply temperature scaling

        outputs = (logits,) + bert_outputs[2:]

        # Calculate loss if labels are provided
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            focal_loss = self.focal_loss_fn(logits, labels)
            hybrid_loss = 0.5 * ce_loss + 0.5 * focal_loss
            outputs = (hybrid_loss,) + outputs

        return outputs
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels, num_numerical_features, class_weights, alpha=0.75, gamma=2.0, *args, **kwargs):
        # Load BERT config from pretrained model or path
        config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels, *args, **kwargs)

        # Create an instance of the class with the provided parameters
        model = cls(
            bert_model_name=model_name_or_path,
            num_labels=num_labels,
            num_numerical_features=num_numerical_features,
            class_weights=class_weights,
            alpha=alpha,
            gamma=gamma,
        )

        # Load pretrained weights and only match compatible ones
        pretrained_state_dict = torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
        model_state_dict = model.state_dict()

        for name, param in pretrained_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].size() == param.size():
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Skipping loading of layer '{name}' due to size mismatch: "
                        f"model layer size {model_state_dict[name].size()} vs. checkpoint layer size {param.size()}")

        model.load_state_dict(model_state_dict, strict=False)

        return model

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.75, gamma=2.0):
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
    data_path = 'training_data_fine_tunePC1.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()
    labels_array = np.array(result)
    model_save_path = 'Fine_Tuned_ModelPC1'

    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")
    model = BertWithNumericalFeatures.from_pretrained(model_save_path, num_labels=2, num_numerical_features=len(numerical_columns), class_weights=class_weights)


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

            outputs = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
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

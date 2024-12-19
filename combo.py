import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, BertConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

class BertWithNumericalFeatures(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, class_weights, alpha=0.75, gamma=1.5):
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        super(BertWithNumericalFeatures, self).__init__(config=bert_model.config)
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate to 0.5 for better regularization
        
        # Simplify model architecture to reduce complexity
        self.fc1 = nn.Linear(self.config.hidden_size, 128)
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, temperature=1.0):
        # BERT model forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[0][:, 0, :]  # Use [CLS] token representation

        # Pass through the additional fully connected layers
        x = self.fc1(cls_output)
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
    def from_pretrained(cls, model_name_or_path, num_labels, class_weights, alpha=0.75, gamma=2.0, *args, **kwargs):
        # Load BERT config from pretrained model or path
        config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels, *args, **kwargs)

        # Create an instance of the class with the provided parameters
        model = cls(
            bert_model_name=model_name_or_path,
            num_labels=num_labels,
            class_weights=class_weights,
            alpha=alpha,
            gamma=gamma,
        )
        
        # Load model state
        model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
        
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

def temperature_scaling(logits, temperature):
    return logits / temperature

def main():
    data_path = 'datasets/mayo_f_lowres_CJ.csv'
    df = pd.read_csv(data_path)

    # Split the DataFrame into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the training and validation DataFrames to CSV files
    train_df.to_csv('datasets/train_dataset.csv', index=False)
    val_df.to_csv('datasets/val_dataset951.csv', index=False)

    print("Training and validation datasets have been saved to 'datasets/train_dataset.csv' and 'datasets/val_dataset.csv' respectively.")

    # Define text and numerical columns
    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Prepare training data
    train_texts = train_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    train_numerical = train_df[numerical_columns].values
    train_labels = train_df.iloc[:, -1].tolist()  # Adjust if label column is not last

    # Prepare validation data
    val_texts = val_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    val_numerical = val_df[numerical_columns].values
    val_labels = val_df.iloc[:, -1].tolist()

    # Proceed with the rest of your code
    model_save_path = 'Mayo_Model951combo'

    labels_array = np.array(train_labels)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")

    # Check if the saved model exists
    if os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
        print("Loading model from saved path...")
        tokenizer = BertTokenizer.from_pretrained(model_save_path)

        model = BertWithNumericalFeatures.from_pretrained(model_save_path, num_labels=2, class_weights=class_weights)
    else:
        print("No saved model found, loading pretrained model from Hugging Face")
        MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        
        # Pass `class_weights` when creating the model
        model = BertWithNumericalFeatures(
            bert_model_name=MODEL_NAME, 
            num_labels=2, 
            class_weights=class_weights,
            alpha=0.75,
            gamma=2.0
        )

    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2

    train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    # Train BERT for categorical features
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # Increased weight decay to 1e-4 for better regularization
    loss_fn = CrossEntropyLoss(weight=class_weights.to(device))

    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)  # Implemented cosine learning rate scheduling

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        accuracy = correct_predictions.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}')

    # Train XGBoost for numerical features
    print("Training XGBoost model with numerical features...")
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=5, learning_rate=0.1)
    xgb_model.fit(train_numerical, train_labels)
    print("XGBoost training complete.")

    # Evaluate XGBoost model
    xgb_preds = xgb_model.predict(val_numerical)
    xgb_accuracy = np.mean(xgb_preds == val_labels)
    print(f'XGBoost Validation Accuracy: {xgb_accuracy}')

def tune_temperature(logits, labels):
    # Find optimal temperature for scaling the logits
    temperature = torch.ones(1, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def loss_fn():
        scaled_logits = logits / temperature
        loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    return temperature.detach()

if __name__ == '__main__':
    main()

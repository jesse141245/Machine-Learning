import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, BertConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

class BertWithNumericalFeatures(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_numerical_features, class_weights=None, alpha=0.6, gamma=1.5):
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        super(BertWithNumericalFeatures, self).__init__(config=bert_model.config)
        self.num_numerical_features = num_numerical_features
        
        # Simplified model structure to reduce overfitting risk
        self.fc1 = nn.Linear(self.config.hidden_size + num_numerical_features, 128)
        self.fc2 = nn.Linear(128, num_labels)
                
        # Activation function
        self.relu = nn.ReLU()
        
        # Batch Normalization layer
        self.bn1 = nn.BatchNorm1d(128)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)  # Dropout rate of 30%

        # Hybrid Loss (Weighted Focal Loss + CrossEntropyLoss)
        if class_weights is not None:
            self.loss_fn_focal = WeightedFocalLoss(class_weights=class_weights, alpha=alpha, gamma=gamma)
            self.loss_fn_ce = CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn_focal = WeightedFocalLoss(alpha=alpha, gamma=gamma)
            self.loss_fn_ce = CrossEntropyLoss()
        
        # Pre-trained BERT model
        self.bert = bert_model.bert

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
        # BERT model forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation

        # Concatenate CLS output with numerical features if available
        if numerical_features is not None:
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
        else:
            combined_output = cls_output

        # Pass through the additional fully connected layers
        x = self.fc1(combined_output)
        x = self.bn1(x)             # Batch Normalization after the first FC layer
        x = self.relu(x)            # Activation Function (ReLU)
        x = self.dropout(x)         # Apply dropout here
        logits = self.fc2(x)

        # Calculate loss if labels are provided (Hybrid Loss)
        loss = None
        if labels is not None:
            focal_loss = self.loss_fn_focal(logits, labels)
            ce_loss = self.loss_fn_ce(logits, labels)
            loss = 0.5 * focal_loss + 0.5 * ce_loss  # Hybrid loss combination

        return loss, logits
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels, num_numerical_features, class_weights=None, alpha=0.6, gamma=1.5, *args, **kwargs):
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
        
        # Load model state
        model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
        
        return model

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

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(numerical, dtype=torch.float),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.6, gamma=1.5):
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

        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(labels.device)  # Move to the same device as labels
            weights = class_weights[labels]
            loss = loss * weights

        return loss.mean()

def main():
    data_path = 'datasets/mayo_az_cleaned_cbert_v2.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=10)  # Reduce to 10 components
    numerical_features = pca.fit_transform(numerical_features)

    # Save the PCA and scaler objects for later use
    with open('pca_transform.pkl', 'wb') as f:
        pickle.dump(pca, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    model_save_path = 'Mayo_Model951'
    train_data_path = 'datasets/train_data851.csv'
    val_data_path = 'val_data951.csv'
    
    labels_array = np.array(result)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")

    if os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
        print("Loading model from saved path...")
        tokenizer = BertTokenizer.from_pretrained(model_save_path)
        model = BertWithNumericalFeatures.from_pretrained(
            model_save_path, 
            num_labels=2, 
            num_numerical_features=pca.n_components_, 
            class_weights=class_weights
        )
    else:
        print("No saved model found, loading pretrained model from Hugging Face")
        MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        
        # Pass class_weights when creating the model
        model = BertWithNumericalFeatures(
            bert_model_name=MODEL_NAME, 
            num_labels=2, 
            num_numerical_features=pca.n_components_,
            class_weights=class_weights,
            alpha=0.6,
            gamma=1.5
        )
        os.makedirs(model_save_path, exist_ok=True)

    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2

    # Split the data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, -1])

    # Save the split data in the original CSV format
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)

    print(f"Training data saved to {train_data_path}")
    print(f"Validation data saved to {val_data_path}")

    train_texts = train_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    val_texts = val_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    train_numerical = scaler.transform(train_df[numerical_columns].values)
    train_numerical = pca.transform(train_numerical)
    val_numerical = scaler.transform(val_df[numerical_columns].values)
    val_numerical = pca.transform(val_numerical)
    train_labels = train_df.iloc[:, -1].tolist()
    val_labels = val_df.iloc[:, -1].tolist()

    train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, MAX_LEN)

    # Use WeightedRandomSampler in DataLoader
    labels = np.array(train_labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weights = 1. / class_sample_count
    samples_weight = np.array([weights[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = model.to(device)

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-5)
    
    num_epochs = 100
    num_training_steps = num_epochs * len(train_loader)
    # Learning rate scheduler
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    best_f1 = 0
    best_precision = 0
    patience = 25
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Pass labels to the model
            loss, logits = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask, labels=labels)

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        accuracy = correct_predictions.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}')

        # Perform validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct_predictions = 0
            all_labels = []
            all_preds = []
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    numerical_features = batch['numerical_features'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    # Pass labels to get loss
                    loss, logits = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask, labels=labels)

                    probs = torch.softmax(logits, dim=1)
                    threshold = 0.3  
                    class_1_probs = probs[:, 1]  
                    preds = (class_1_probs > threshold).long()

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='binary')
            val_recall = recall_score(all_labels, all_preds, average='binary')
            val_f1 = f1_score(all_labels, all_preds, average='binary')

            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}')

            # Early stopping based on F1 and Precision
            if val_loss < best_val_loss or val_f1 > best_f1 or val_precision > best_precision:
                best_val_loss = min(val_loss, best_val_loss)
                best_f1 = max(val_f1, best_f1)
                best_precision = max(val_precision, best_precision)
                patience_counter = 0
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f'Model and tokenizer saved to {model_save_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in F1 and Precision.")
                    break

if __name__ == '__main__':
    main()

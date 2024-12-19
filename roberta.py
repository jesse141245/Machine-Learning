import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

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

def main():
    data_path = 'datasets/mayo_az_pred.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()

    model_save_path = 'Roberta_Model'
    train_data_path = 'datasets/train_data.csv'
    val_data_path = 'datasets/val_data.csv'
    
    labels_array = np.array(result)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaWithNumericalFeatures(
        roberta_model_name='roberta-base', 
        num_labels=2, 
        num_numerical_features=len(numerical_columns),
        class_weights=class_weights,
        alpha=0.75,
        gamma=2.0
    )

    MAX_LEN = 256
    BATCH_SIZE = 16  # Reduced batch size for smaller dataset
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
    train_numerical = train_df[numerical_columns].values
    val_numerical = val_df[numerical_columns].values
    train_labels = train_df.iloc[:, -1].tolist()
    val_labels = val_df.iloc[:, -1].tolist()

    train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, MAX_LEN)

    # Implement WeightedRandomSampler
    labels = np.array(train_labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weights = 1. / class_sample_count
    samples_weight = np.array([weights[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)

    num_epochs = 50
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    best_f1 = 0
    best_precision = 0
    patience = 10  # Adjusted patience for early stopping
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask, labels=labels)

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        accuracy = correct_predictions.double() / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy:.4f}")

        # Adjust learning rate, alpha, and gamma based on training results
        if accuracy < 0.7:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.9, 1e-6)  # Reduce learning rate if accuracy is low
            model.loss_fn.alpha = min(model.loss_fn.alpha + 0.05, 1.0)  # Increase alpha to focus more on minority class
            model.loss_fn.gamma = min(model.loss_fn.gamma + 0.1, 5.0)  # Increase gamma to penalize easy examples more
        elif accuracy > 0.85:
            for param_group in optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 1e-3)  # Increase learning rate if accuracy is high
            model.loss_fn.alpha = max(model.loss_fn.alpha - 0.05, 0.5)  # Decrease alpha to reduce focus on minority class
            model.loss_fn.gamma = max(model.loss_fn.gamma - 0.1, 1.0)  # Decrease gamma to reduce penalty on easy examples

        # Validation phase
        model.eval()
        all_labels = []
        all_preds = []
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)

                # Using threshold of 0.5 for binary classification
                class_1_probs = probs[:, 1]
                threshold = 0.5 - (epoch * 0.005) if epoch < 10 else 0.45  # Adjust threshold over epochs
                preds = (class_1_probs > threshold).long()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                loss = model.loss_fn(logits.view(-1, 2), labels.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='binary')
        val_recall = recall_score(all_labels, all_preds, average='binary')
        val_f1 = f1_score(all_labels, all_preds, average='binary')

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

        # Early stopping based on F1 and precision
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
        model.roberta.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model and tokenizer saved to {model_save_path}")
        if val_loss < best_val_loss or val_f1 > best_f1 or val_precision > best_precision:
            best_val_loss = min(val_loss, best_val_loss)
            best_f1 = max(val_f1, best_f1)
            best_precision = max(val_precision, best_precision)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in F1 and Precision.")
                break

if __name__ == '__main__':
    main()

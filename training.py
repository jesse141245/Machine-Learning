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
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class BertWithNumericalFeatures(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_numerical_features):
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        super(BertWithNumericalFeatures, self).__init__(config=bert_model.config)
        self.num_numerical_features = num_numerical_features
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.config.hidden_size + num_numerical_features, num_labels)
        self.bert = bert_model.bert

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[0][:, 0, :]  # Use [CLS] token representation

        if numerical_features is not None:
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
            combined_output = self.dropout(combined_output)
        else:
            combined_output = cls_output

        logits = self.fc(combined_output)
        outputs = (logits,) + bert_outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels, num_numerical_features, *args, **kwargs):
        config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels, *args, **kwargs)
        
        model = cls(bert_model_name=model_name_or_path, num_labels=num_labels, num_numerical_features=num_numerical_features)
        
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
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)  # Probability for the true label
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

def main():
    data_path = 'datasets/selected_features_orig.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()

    model_save_path = 'selected_orig_model'
    labels_array = np.array(result)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")

    if os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
        print("Loading model from saved path...")
        tokenizer = BertTokenizer.from_pretrained(model_save_path)
        model = BertWithNumericalFeatures.from_pretrained(model_save_path, num_labels=2, num_numerical_features=len(numerical_columns))
    else:
        print("No saved model found, loading pretrained model from Hugging Face")
        MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertWithNumericalFeatures(MODEL_NAME, num_labels=2, num_numerical_features=len(numerical_columns))
        os.makedirs(model_save_path, exist_ok=True)

    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2

    train_texts, val_texts, train_numerical, val_numerical, train_labels, val_labels = train_test_split(
        texts, numerical_features, result, test_size=0.2, random_state=42
    )

    train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-5)
    loss_fn = CrossEntropyLoss(weight=class_weights.to(device))

    num_epochs = 100
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    best_f1 = 0
    best_precision = 0
    patience = 5
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

            outputs = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask)
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

                outputs = model(input_ids=input_ids, numerical_features=numerical_features, attention_mask=attention_mask)
                logits = outputs[0]
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
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

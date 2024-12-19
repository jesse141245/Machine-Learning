import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, BertConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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
        cls_output = bert_outputs[0][:, 0, :]  

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

def tune_threshold(preds, labels):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        thresholded_preds = (preds >= threshold).astype(int)
        f1 = f1_score(labels, thresholded_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

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
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

def main():
    data_path = 'datasets/selected_features_orig.csv'
    df = pd.read_csv(data_path)

    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    labels = df.iloc[:, -1].values  

    model_save_path = 'selected_orig_model1'

    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Computed class weights: {class_weights}")

    skf = StratifiedKFold(n_splits=5)
    all_preds = []
    all_labels = []

    for train_index, val_index in skf.split(numerical_features, labels):
        train_texts, val_texts = [texts[i] for i in train_index], [texts[i] for i in val_index]
        train_numerical, val_numerical = numerical_features[train_index], numerical_features[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        smote = SMOTE(sampling_strategy=0.8, k_neighbors=5, random_state=42)
        smote_numerical, smote_labels = smote.fit_resample(train_numerical, train_labels)

        mean = 0
        std_dev = 0.1
        noise = np.random.normal(mean, std_dev, smote_numerical.shape)
        smote_numerical += noise

        text_multiplier = len(smote_labels) // len(train_texts) + 1
        smote_texts = (train_texts * text_multiplier)[:len(smote_labels)]

        train_texts = smote_texts
        train_numerical = smote_numerical
        train_labels = smote_labels

        tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, max_len=256)
        val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, max_len=256)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = BertWithNumericalFeatures('emilyalsentzer/Bio_ClinicalBERT', num_labels=2, num_numerical_features=len(numerical_columns))
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        loss_fn = FocalLoss(alpha=class_weights[1], gamma=2).to(device)

        num_epochs = 10
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_loader))

        best_val_loss = float('inf')
        best_f1 = 0
        patience = 3
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

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

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

            model.eval()
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
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = (probs >= 0.5).astype(int)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds)

                    loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='binary')
            val_recall = recall_score(all_labels, all_preds, average='binary')
            val_f1 = f1_score(all_labels, all_preds, average='binary')
            val_auc = roc_auc_score(all_labels, all_preds)

            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}, AUC: {val_auc:.4f}')

            if val_loss < best_val_loss or val_f1 > best_f1:
                best_val_loss = min(val_loss, best_val_loss)
                best_f1 = max(val_f1, best_f1)
                patience_counter = 0

                torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f'Model and tokenizer saved to {model_save_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in F1.")
                    break

if __name__ == '__main__':
    main()

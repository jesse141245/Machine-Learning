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
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class BertWithNumericalFeatures(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_numerical_features, class_weights, alpha=0.75, gamma=1.5):
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        super(BertWithNumericalFeatures, self).__init__(config=bert_model.config)

        # Set attributes to make it compatible with sklearn
        self.bert_model_name = bert_model_name
        self.num_labels = num_labels
        self.num_numerical_features = num_numerical_features
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

        # Increase model complexity
        self.num_numerical_features = num_numerical_features
        self.dropout = nn.Dropout(p=0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.config.hidden_size + num_numerical_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_labels)
                
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU(alpha=1.0)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        # Initialize WeightedFocalLoss with class_weights
        self.loss_fn = WeightedFocalLoss(class_weights=class_weights, alpha=alpha, gamma=gamma)
        
        # Pre-trained BERT model
        self.bert = bert_model.bert
        
    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
    # BERT model forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.pooler_output


        # Concatenate CLS output with numerical features if available
        if numerical_features is not None:
            if len(numerical_features.shape) == 1:
                numerical_features = numerical_features.unsqueeze(0)  # Ensure batch dimension

            # Concatenate CLS output with numerical features if available
            combined_output = torch.cat((cls_output, numerical_features), dim=1)
            combined_output = self.dropout(combined_output)
        else:
            combined_output = cls_output

        # Pass through the additional fully connected layers
        x = self.fc1(combined_output)
        x = self.bn1(x)             # Batch Normalization after the first FC layer
        x = self.leaky_relu(x)      # Activation Function (LeakyReLU)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)             # Batch Normalization after the second FC layer
        x = self.elu(x)             # Activation Function (ELU)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        logits = self.fc5(x)

        outputs = (logits,) + bert_outputs[2:]

        # Calculate loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

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
            'numerical_features': torch.tensor(numerical, dtype=torch.float)
        }

        if self.labels is not None:
            label = self.labels[idx]
            item['labels'] = torch.tensor(label, dtype=torch.long)

        return item


    def get_numpy_features_and_labels(self):
        all_features = []
        all_labels = []

        for idx in range(len(self.texts)):
            item = self.__getitem__(idx)
            input_ids = item['input_ids'].numpy()  # Convert input_ids to numpy
            attention_mask = item['attention_mask'].numpy()  # Convert attention_mask to numpy
            numerical_features = item['numerical_features'].squeeze(0).numpy()  # Convert numerical features to numpy

            # Flatten the input_ids and attention_mask and concatenate with numerical features
            features = np.concatenate((input_ids.flatten(), attention_mask.flatten(), numerical_features))
            all_features.append(features)
            all_labels.append(item['labels'].item())

        return np.array(all_features), np.array(all_labels)

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.65, gamma=2.0):
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
class SklearnBertWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, tokenizer, num_epochs=8, batch_size=32, max_len=256, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X, y):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=2e-5)
        loss_fn = self.model.loss_fn

        # Create a Dataset and DataLoader from X and y
        dataset = CustomDataset(X['texts'], X['numerical_features'], y, self.tokenizer, self.max_len)
        class_sample_count = np.array([len(np.where(dataset == t)[0]) for t in np.unique(dataset)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in dataset])

        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # Use sampler in DataLoader
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
                logits = outputs[0]
                loss = loss_fn(logits.view(-1, self.model.num_labels), labels.view(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict(self, X):
        self.model.eval()
        dataset = CustomDataset(X['texts'], X['numerical_features'], None, self.tokenizer, self.max_len)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
                logits = outputs[0]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)
def manual_bagging_train(train_features, train_labels, tokenizer, model_params, num_estimators=10):
    models = []
    for i in range(num_estimators):
        print(f"Training model {i+1}/{num_estimators}")
        # Create a bootstrap sample
        indices = np.random.choice(len(train_labels), size=len(train_labels), replace=True)
        bootstrap_texts = [train_features['texts'][idx] for idx in indices]
        bootstrap_numerical = train_features['numerical_features'][indices]
        bootstrap_labels = [train_labels[idx] for idx in indices]

        # Prepare the dataset
        bootstrap_dataset = CustomDataset(
            texts=bootstrap_texts,
            numerical_features=bootstrap_numerical,
            labels=bootstrap_labels,
            tokenizer=tokenizer,
            max_len=model_params['max_len']
        )
        train_loader = DataLoader(bootstrap_dataset, batch_size=model_params['batch_size'], shuffle=True)

        # Initialize a new model
        model = BertWithNumericalFeatures(
            bert_model_name=model_params['bert_model_name'],
            num_labels=model_params['num_labels'],
            num_numerical_features=model_params['num_numerical_features'],
            class_weights=model_params['class_weights'],
            alpha=model_params['alpha'],
            gamma=model_params['gamma']
        )
        model.to(model_params['device'])

        # Train the model
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_params['learning_rate'], weight_decay=2e-5)
        loss_fn = model.loss_fn

        model.train()
        for epoch in range(model_params['num_epochs']):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(model_params['device'])
                attention_mask = batch['attention_mask'].to(model_params['device'])
                numerical_features = batch['numerical_features'].to(model_params['device'])
                labels = batch['labels'].to(model_params['device'])

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features
                )
                logits = outputs[0]
                loss = loss_fn(logits.view(-1, model.num_labels), labels.view(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        models.append(model)
    return models

def manual_bagging_predict(models, val_features, tokenizer, model_params):
    val_dataset = CustomDataset(
        texts=val_features['texts'],
        numerical_features=val_features['numerical_features'],
        labels=None,  # labels are None during prediction
        tokenizer=tokenizer,
        max_len=model_params['max_len']
    )
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=False)

    all_model_preds = []

    for model in models:
        model.eval()
        model_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model_params['device'])
                attention_mask = batch['attention_mask'].to(model_params['device'])
                numerical_features = batch['numerical_features'].to(model_params['device'])

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features
                )
                logits = outputs[0]
                preds = torch.argmax(logits, dim=1)
                model_preds.extend(preds.cpu().numpy())
        all_model_preds.append(model_preds)

    # Aggregate predictions using majority voting
    aggregated_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(all_model_preds)
    )
    return aggregated_preds



def prepare_dataset():
    # Load the data
    data_path = 'datasets/mayo_az_pred.csv'
    df = pd.read_csv(data_path)

    # Define text and numerical columns
    text_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    texts = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    numerical_features = df[numerical_columns].values
    result = df.iloc[:, -1].tolist()

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, -1])
    print("Data split into training and validation sets.")

    # Save the training and validation sets to CSV files
    train_data_path = 'datasets/train_data.csv'
    val_data_path = 'datasets/val_dataset.csv'
    
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    
    print(f"Training data saved to {train_data_path}")
    print(f"Validation data saved to {val_data_path}")

    return train_df, val_df, text_columns, numerical_columns

def setup_model_and_tokenizer(text_columns, numerical_columns):
    model_save_path = 'Mayo_Model3'

    # Load the tokenizer
    if os.path.exists(model_save_path):
        tokenizer = BertTokenizer.from_pretrained(model_save_path)
    else:
        MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Load or create model
    class_weights = [0.5, 2.0]  # Replace with computed class weights from your data if available
    if os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
        print("Loading model from saved path...")
        model = BertWithNumericalFeatures.from_pretrained(model_save_path, num_labels=2, num_numerical_features=len(numerical_columns), class_weights=class_weights)
    else:
        print("No saved model found, loading pretrained model from Hugging Face")
        model = BertWithNumericalFeatures(
            bert_model_name='emilyalsentzer/Bio_ClinicalBERT', 
            num_labels=2, 
            num_numerical_features=len(numerical_columns), 
            class_weights=class_weights
        )

    return model, tokenizer

def setup_bagging_model(model, tokenizer):
    # Wrap your custom BERT model
    wrapped_model = SklearnBertWrapper(model=model, tokenizer=tokenizer)

    # Define BaggingClassifier with your custom model as the base estimator
    bagging_model = BaggingClassifier(
        estimator=wrapped_model,  # Use 'estimator' instead of 'base_estimator'
        n_estimators=5,  # Number of models in the ensemble
        random_state=42,
        n_jobs=-1  # Use all processors to train models in parallel
    )
    return bagging_model

def train_and_evaluate(model, bagging_model, train_df, val_df, tokenizer, text_columns, numerical_columns):
    # Prepare training and validation datasets
    MAX_LEN = 256
    train_texts = train_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    val_texts = val_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    train_numerical = train_df[numerical_columns].values
    val_numerical = val_df[numerical_columns].values
    train_labels = train_df.iloc[:, -1].tolist()
    val_labels = val_df.iloc[:, -1].tolist()

    # Create datasets
    train_dataset = CustomDataset(train_texts, train_numerical, train_labels, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_numerical, val_labels, tokenizer, MAX_LEN)

    # Fit BaggingClassifier on the training dataset
    print("Fitting bagging model...")
    bagging_model.fit(train_dataset, train_labels)

    # Predict on validation dataset
    print("Evaluating model on validation set...")
    y_pred = bagging_model.predict(val_dataset)

    # Evaluate metrics
    accuracy = accuracy_score(val_labels, y_pred)
    precision = precision_score(val_labels, y_pred)
    recall = recall_score(val_labels, y_pred)
    f1 = f1_score(val_labels, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")


def main():
    # Prepare the dataset
    train_df, val_df, text_columns, numerical_columns = prepare_dataset()

    # Set up the model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(text_columns, numerical_columns)

    # Prepare training data
    train_texts = train_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    train_numerical = train_df[numerical_columns].values
    train_labels = train_df.iloc[:, -1].tolist()

    # Create a dictionary of features
    train_features = {
        'texts': train_texts,
        'numerical_features': train_numerical
    }

    # Prepare validation data
    val_texts = val_df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    val_numerical = val_df[numerical_columns].values
    val_labels = val_df.iloc[:, -1].tolist()

    val_features = {
        'texts': val_texts,
        'numerical_features': val_numerical
    }

    # Define model parameters
    model_params = {
        'bert_model_name': model.bert_model_name,
        'num_labels': model.num_labels,
        'num_numerical_features': model.num_numerical_features,
        'class_weights': model.class_weights,
        'alpha': model.alpha,
        'gamma': model.gamma,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_len': 256,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'num_epochs': 3  # Adjust as needed
    }

    # Train ensemble models
    models = manual_bagging_train(train_features, train_labels, tokenizer, model_params, num_estimators=8)

    # Predict on validation data
    y_pred = manual_bagging_predict(models, val_features, tokenizer, model_params)

    # Evaluate metrics
    accuracy = accuracy_score(val_labels, y_pred)
    precision = precision_score(val_labels, y_pred, zero_division=0)
    recall = recall_score(val_labels, y_pred, zero_division=0)
    f1 = f1_score(val_labels, y_pred, zero_division=0)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    model_save_path = 'Mayo_Model3'  # You can modify this to specify your desired save path

    # Ensure the directory exists
    os.makedirs(model_save_path, exist_ok=True)

    # Save the model state
    torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f'Model and tokenizer saved to {model_save_path}')
if __name__ == '__main__':
    main()

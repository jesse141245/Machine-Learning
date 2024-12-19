import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    BertPreTrainedModel,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
)
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
seed = random.randint(0, 10000)  # Generate a random seed for every run
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class BertWithNumericalFeatures(BertPreTrainedModel):
    def __init__(self, config, class_weights=None, **kwargs):
        super(BertWithNumericalFeatures, self).__init__(config)
        self.num_numerical_features = config.num_numerical_features
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.num_labels = config.num_labels
        self.class_weights = class_weights

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

        # Initialize WeightedFocalLoss with class_weights
        self.loss_fn = WeightedFocalLoss(
            class_weights=self.class_weights, alpha=self.alpha, gamma=self.gamma
        )

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
            numerical_features = numerical_features.to(cls_output.device)
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
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return loss, logits


class CustomDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_len):
        self.texts = texts
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Standardize numerical features if they exist
        if self.numerical_features is not None and self.numerical_features.shape[1] > 0:
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


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.75, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(
            labels, num_classes=logits.size(1)
        ).float()
        pt = torch.sum(probs * labels_one_hot, dim=1)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return loss.mean()


# Temperature Scaling Module
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        numerical_features=None,
        labels=None,
    ):
        loss, logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_features=numerical_features,
            labels=labels,
        )

        # Apply temperature scaling
        logits = self.temperature_scale(logits)
        return loss, logits

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Collecting logits for temperature scaling"):
                input_ids = batch['input_ids'].to(self.temperature.device)
                attention_mask = batch['attention_mask'].to(self.temperature.device)
                numerical_features = batch['numerical_features'].to(self.temperature.device) if batch['numerical_features'] is not None else None
                labels = batch['labels'].to(self.temperature.device)

                _, logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    labels=labels,
                )
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss().to(self.temperature.device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f'Optimal temperature: {self.temperature.item()}')

    def predict(self, input_ids, attention_mask, numerical_features):
        with torch.no_grad():
            _, logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                labels=None,
            )
            probs = torch.softmax(logits, dim=1)
        return probs


def main():
    # Common configurations
    MAX_LEN = 256
    BATCH_SIZE = 32
    num_workers = os.cpu_count() // 2 or 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Paths for saved models and data
    initial_model_save_path = "Initial_Model"
    fine_tuned_model_save_path = "Mayo_ModelF"
    training_data_fine_tune_csv = "training_data_fine_tuneF.csv"
    validation_data_fine_tune_csv = "validation_data_fine_tuneF.csv"

    # Load Datasets and Prepare Numerical Features

    # Load datasets
    initial_data_path = "datasets/rejected.csv"
    fine_tune_data_path = "datasets/mayo_f_lowres_CJ.csv"
    df_initial = pd.read_csv(initial_data_path)
    df_fine_tune = pd.read_csv(fine_tune_data_path)

    # Placeholder for column renaming if needed
    column_renaming = {
        # 'OldColumnNameInFineTune': 'MatchingColumnNameInInitial',
    }
    df_fine_tune.rename(columns=column_renaming, inplace=True)

    # Placeholder for numerical columns conversion if needed
    numerical_cols_to_convert = []  # Replace with your column names if necessary
    for col in numerical_cols_to_convert:
        if col in df_initial.columns:
            df_initial[col] = pd.to_numeric(df_initial[col], errors='coerce')
        if col in df_fine_tune.columns:
            df_fine_tune[col] = pd.to_numeric(df_fine_tune[col], errors='coerce')

    numerical_columns_initial = df_initial.select_dtypes(include=["float64", "int64"]).columns.tolist()
    print(f"Numerical columns in initial dataset ({len(numerical_columns_initial)}): {numerical_columns_initial}")

    numerical_columns_fine_tune = df_fine_tune.select_dtypes(include=["float64", "int64"]).columns.tolist()
    print(f"Numerical columns in fine-tuning dataset ({len(numerical_columns_fine_tune)}): {numerical_columns_fine_tune}")

    text_columns_initial = df_initial.select_dtypes(include=["object"]).columns.tolist()
    text_columns_fine_tune = df_fine_tune.select_dtypes(include=["object"]).columns.tolist()

    num_labels = 2
    alpha = 0.75
    gamma = 2.0

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    config = BertConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    ##### Initial Training Phase #####

    # Check if initial model is already saved
    if os.path.exists(initial_model_save_path):
        print(f"Initial model found at {initial_model_save_path}, skipping initial training.")
    else:
        print("Starting initial training...")
        # Prepare data (texts, numerical features, labels)
        texts_initial = df_initial[text_columns_initial].apply(
            lambda x: " ".join(x.astype(str)), axis=1
        ).tolist()
        num_numerical_features_initial = len(numerical_columns_initial)
        config.num_numerical_features = num_numerical_features_initial
        config.alpha = alpha
        config.gamma = gamma

        if num_numerical_features_initial > 0:
            numerical_features_initial = df_initial[numerical_columns_initial].values
        else:
            numerical_features_initial = np.empty((len(df_initial), 0))
        labels_initial = df_initial.iloc[:, -1].tolist()  # Assuming label is the last column

        # Compute class weights
        labels_array_initial = np.array(labels_initial)
        class_weights_initial = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels_array_initial), y=labels_array_initial
        )
        class_weights_initial = torch.tensor(class_weights_initial, dtype=torch.float)
        print(f"Computed class weights for initial training: {class_weights_initial}")

        # Initialize the model with the config and class_weights
        model = BertWithNumericalFeatures(config, class_weights=class_weights_initial)
        model.to(device)

        # Split the data
        train_df_initial, val_df_initial = train_test_split(
            df_initial, test_size=0.2, random_state=seed, stratify=labels_initial
        )

        # Save validation dataset to CSV
        val_df_initial.to_csv('validation_data_initial.csv', index=False)
        print("Validation dataset for initial training saved to 'validation_data_initial.csv'.")

        # Prepare datasets
        train_texts_initial = train_df_initial[text_columns_initial].apply(
            lambda x: " ".join(x.astype(str)), axis=1
        ).tolist()
        val_texts_initial = val_df_initial[text_columns_initial].apply(
            lambda x: " ".join(x.astype(str)), axis=1
        ).tolist()
        if num_numerical_features_initial > 0:
            train_numerical_initial = train_df_initial[numerical_columns_initial].values
            val_numerical_initial = val_df_initial[numerical_columns_initial].values
        else:
            train_numerical_initial = np.empty((len(train_df_initial), 0))
            val_numerical_initial = np.empty((len(val_df_initial), 0))
        train_labels_initial = train_df_initial.iloc[:, -1].tolist()
        val_labels_initial = val_df_initial.iloc[:, -1].tolist()

        # Create datasets
        train_dataset_initial = CustomDataset(
            train_texts_initial,
            train_numerical_initial,
            train_labels_initial,
            tokenizer,
            MAX_LEN,
        )
        val_dataset_initial = CustomDataset(
            val_texts_initial,
            val_numerical_initial,
            val_labels_initial,
            tokenizer,
            MAX_LEN,
        )

        # Create DataLoaders with WeightedRandomSampler
        labels_train_initial = np.array(train_labels_initial)
        class_sample_count_initial = np.array(
            [len(np.where(labels_train_initial == t)[0]) for t in np.unique(labels_train_initial)]
        )
        weights_initial = 1.0 / class_sample_count_initial
        samples_weight_initial = np.array([weights_initial[t] for t in labels_train_initial])
        sampler_initial = WeightedRandomSampler(
            samples_weight_initial, len(samples_weight_initial)
        )

        train_loader_initial = DataLoader(
            train_dataset_initial,
            batch_size=BATCH_SIZE,
            sampler=sampler_initial,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader_initial = DataLoader(
            val_dataset_initial,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Set up training configurations
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=2e-5)
        num_epochs = 20  # Adjust the number of epochs as needed
        num_training_steps = num_epochs * len(train_loader_initial)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Training loop
        best_val_loss = float("inf")
        best_f1 = 0
        best_precision = 0
        patience = 15  # Adjust patience for early stopping
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct_predictions = 0

            for batch in tqdm(
                train_loader_initial, desc=f"Initial Training Epoch {epoch + 1}/{num_epochs}"
            ):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                numerical_features = (
                    batch["numerical_features"].to(device)
                    if batch["numerical_features"] is not None
                    else None
                )

                loss, logits = model(
                    input_ids=input_ids,
                    numerical_features=numerical_features,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

            accuracy = correct_predictions.double() / len(train_loader_initial.dataset)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader_initial):.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )

            # Validation phase
            model.eval()
            all_labels = []
            all_preds = []
            val_loss = 0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader_initial, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"
                ):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    numerical_features = (
                        batch["numerical_features"].to(device)
                        if batch["numerical_features"] is not None
                        else None
                    )

                    loss, logits = model(
                        input_ids=input_ids,
                        numerical_features=numerical_features,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    class_1_probs = probs[:, 1]
                    preds = (class_1_probs > 0.5).long()

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            val_loss /= len(val_loader_initial)
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_recall = recall_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

            print(
                f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}"
            )

            # Early stopping based on F1 and precision
            if val_loss < best_val_loss or val_f1 > best_f1 or val_precision > best_precision:
                best_val_loss = min(val_loss, best_val_loss)
                best_f1 = max(val_f1, best_f1)
                best_precision = max(val_precision, best_precision)
                patience_counter = 0
                # Save the best model
                if not os.path.exists(initial_model_save_path):
                    os.makedirs(initial_model_save_path)
                model.save_pretrained(initial_model_save_path)
                tokenizer.save_pretrained(initial_model_save_path)
                print(f"Initial model and tokenizer saved to {initial_model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs due to no improvement."
                    )
                    break

    ##### Fine-tuning Phase #####

    # Check if fine-tuned model and training data are already saved
    if os.path.exists(fine_tuned_model_save_path) and os.path.exists(training_data_fine_tune_csv):
        print(f"Fine-tuned model and training data found, loading them to continue training.")

        # Load the fine-tuned model
        config_fine_tune = BertConfig.from_pretrained(fine_tuned_model_save_path)

        # Add custom attributes
        config_fine_tune.num_labels = num_labels
        config_fine_tune.num_numerical_features = len(numerical_columns_fine_tune)  # Set the number of numerical features
        config_fine_tune.alpha = alpha
        config_fine_tune.gamma = gamma

        model_fine_tune = BertWithNumericalFeatures.from_pretrained(
            fine_tuned_model_save_path,
            config=config_fine_tune,
            class_weights=None  # You can adjust class_weights if needed
        )

        model_fine_tune.to(device)

        # Load the training and validation datasets
        train_df_fine_tune = pd.read_csv(training_data_fine_tune_csv)
        val_df_fine_tune = pd.read_csv(validation_data_fine_tune_csv)
        print(f"Loaded training and validation data from CSV files.")

    else:
        print("Starting fine-tuning...")
        # Split the data
        train_df_fine_tune, val_df_fine_tune = train_test_split(
            df_fine_tune, test_size=0.3, random_state=seed, stratify=df_fine_tune.iloc[:, -1]
        )

        # Save training and validation datasets to CSV
        train_df_fine_tune.to_csv(training_data_fine_tune_csv, index=False)
        val_df_fine_tune.to_csv(validation_data_fine_tune_csv, index=False)
        print(f"Training and validation datasets for fine-tuning saved to '{training_data_fine_tune_csv}' and '{validation_data_fine_tune_csv}'.")

        # Initialize the model for fine-tuning
        initial_model_save_path = "Initial_Model"
        config_fine_tune = BertConfig.from_pretrained(initial_model_save_path)
        num_numerical_features_fine_tune = len(numerical_columns_fine_tune)
        config_fine_tune.num_labels = num_labels
        config_fine_tune.num_numerical_features = num_numerical_features_fine_tune
        config_fine_tune.alpha = alpha
        config_fine_tune.gamma = gamma

        # Compute class weights
        labels_fine_tune = df_fine_tune.iloc[:, -1].tolist()
        labels_array_fine_tune = np.array(labels_fine_tune)
        class_weights_fine_tune = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels_array_fine_tune),
            y=labels_array_fine_tune,
        )
        class_weights_fine_tune = torch.tensor(class_weights_fine_tune, dtype=torch.float)
        print(
            f"Computed class weights for fine-tuning: {class_weights_fine_tune}"
        )

        # Load the pre-trained model using from_pretrained with ignore_mismatched_sizes=True
        model_fine_tune = BertWithNumericalFeatures.from_pretrained(
            initial_model_save_path,
            config=config_fine_tune,
            class_weights=class_weights_fine_tune,
            ignore_mismatched_sizes=True
        )
        model_fine_tune.to(device)
        print("Loaded pre-trained weights into the fine-tuning model.")

    # Prepare datasets
    train_texts_fine_tune = train_df_fine_tune[text_columns_fine_tune].apply(
        lambda x: " ".join(x.astype(str)), axis=1
    ).tolist()
    val_texts_fine_tune = val_df_fine_tune[text_columns_fine_tune].apply(
        lambda x: " ".join(x.astype(str)), axis=1
    ).tolist()
    if len(numerical_columns_fine_tune) > 0:
        train_numerical_fine_tune = train_df_fine_tune[numerical_columns_fine_tune].values
        val_numerical_fine_tune = val_df_fine_tune[numerical_columns_fine_tune].values
    else:
        train_numerical_fine_tune = np.empty((len(train_df_fine_tune), 0))
        val_numerical_fine_tune = np.empty((len(val_df_fine_tune), 0))
    train_labels_fine_tune = train_df_fine_tune.iloc[:, -1].tolist()
    val_labels_fine_tune = val_df_fine_tune.iloc[:, -1].tolist()

    # Create datasets
    train_dataset_fine_tune = CustomDataset(
        train_texts_fine_tune,
        train_numerical_fine_tune,
        train_labels_fine_tune,
        tokenizer,
        MAX_LEN,
    )
    val_dataset_fine_tune = CustomDataset(
        val_texts_fine_tune,
        val_numerical_fine_tune,
        val_labels_fine_tune,
        tokenizer,
        MAX_LEN,
    )

    # Create DataLoaders with WeightedRandomSampler
    labels_train_fine_tune = np.array(train_labels_fine_tune)
    class_sample_count_fine_tune = np.array(
        [len(np.where(labels_train_fine_tune == t)[0]) for t in np.unique(labels_train_fine_tune)]
    )
    weights_fine_tune = 1.0 / class_sample_count_fine_tune
    samples_weight_fine_tune = np.array(
        [weights_fine_tune[t] for t in labels_train_fine_tune]
    )
    sampler_fine_tune = WeightedRandomSampler(
        samples_weight_fine_tune, len(samples_weight_fine_tune)
    )

    train_loader_fine_tune = DataLoader(
        train_dataset_fine_tune,
        batch_size=BATCH_SIZE,
        sampler=sampler_fine_tune,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader_fine_tune = DataLoader(
        val_dataset_fine_tune,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Set up fine-tuning configurations
    optimizer = torch.optim.AdamW(model_fine_tune.parameters(), lr=5e-6, weight_decay=2e-5)
    num_epochs = 50  # Adjust the number of epochs as needed
    num_training_steps = num_epochs * len(train_loader_fine_tune)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Fine-tuning loop
    best_val_loss = float("inf")
    best_f1 = 0
    best_precision = 0
    patience = 50  # Adjust patience for early stopping
    patience_counter = 0

    for epoch in range(num_epochs):
        model_fine_tune.train()
        total_loss = 0
        correct_predictions = 0

        for batch in tqdm(
            train_loader_fine_tune, desc=f"Fine-tuning Epoch {epoch + 1}/{num_epochs}"
        ):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            numerical_features = (
                batch["numerical_features"].to(device)
                if batch["numerical_features"] is not None
                else None
            )

            loss, logits = model_fine_tune(
                input_ids=input_ids,
                numerical_features=numerical_features,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_fine_tune.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        accuracy = correct_predictions.double() / len(train_loader_fine_tune.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader_fine_tune):.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

        # Validation phase
        if (epoch + 1) % 10 == 0:
            model_fine_tune.eval()
            all_labels = []
            all_preds = []
            val_loss = 0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader_fine_tune, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"
                ):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    numerical_features = (
                        batch["numerical_features"].to(device)
                        if batch["numerical_features"] is not None
                        else None
                    )

                    loss, logits = model_fine_tune(
                        input_ids=input_ids,
                        numerical_features=numerical_features,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    class_1_probs = probs[:, 1]
                    preds = (class_1_probs > 0.5).long()

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            val_loss /= len(val_loader_fine_tune)
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_recall = recall_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

            print(
                f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}"
            )

            # Early stopping based on F1 and precision
            if val_loss < best_val_loss or val_f1 > best_f1 or val_precision > best_precision:
                best_val_loss = min(val_loss, best_val_loss)
                best_f1 = max(val_f1, best_f1)
                best_precision = max(val_precision, best_precision)
                patience_counter = 0
                # Save the best model
                if not os.path.exists(fine_tuned_model_save_path):
                    os.makedirs(fine_tuned_model_save_path)
                model_fine_tune.save_pretrained(fine_tuned_model_save_path)
                tokenizer.save_pretrained(fine_tuned_model_save_path)
                print(f"Fine-tuned model and tokenizer saved to {fine_tuned_model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs due to no improvement."
                    )
                    break

            # Evaluate with calibrated model
            model_fine_tune.eval()
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for batch in tqdm(
                    val_loader_fine_tune, desc=f"Evaluating Calibrated Model Epoch {epoch + 1}/{num_epochs}"
                ):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    numerical_features = (
                        batch["numerical_features"].to(device)
                        if batch["numerical_features"] is not None
                        else None
                    )

                    _, logits = model_fine_tune(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        numerical_features=numerical_features,
                        labels=None,
                    )

                    probs = torch.softmax(logits, dim=1)
                    class_1_probs = probs[:, 1]
                    preds = (class_1_probs > 0.5).long()

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_recall = recall_score(
                all_labels, all_preds, average="binary", zero_division=0
            )
            val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

            print(
                f"Calibrated Model - Accuracy: {val_accuracy:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}"
            )


if __name__ == "__main__":
    main()

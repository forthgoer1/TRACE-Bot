import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
)
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class SocialBotDataset(Dataset):
    def __init__(self, texts, behaviors, labels, tokenizer, max_len=256):
        self.texts = texts
        self.behaviors = behaviors
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'behavior': torch.tensor(self.behaviors[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class GPT2SocialBotDetector(nn.Module):
    def __init__(self, gpt2_model_name='gpt2', behavior_dim=47, gpt2_emb_dim=768, hidden_dim=128):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)

        self.mlp_behavior = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fuse = nn.Sequential(
            nn.Linear(gpt2_emb_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_ids, attention_mask, behavior):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        gpt2_emb = last_hidden[batch_indices, seq_lengths]

        behav_emb = self.mlp_behavior(behavior)
        fused = torch.cat([gpt2_emb, behav_emb], dim=1)
        logits = self.fuse(fused)
        return logits

def evaluate_metrics(model, data_loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            behavior = batch['behavior'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, behavior)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prc = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc = roc_auc_score(all_labels, all_probs)
    bep = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prc, rec, f1, roc, bep, cm

def train_model(input_file, model_save_path):
    df = pd.read_csv(input_file)
    y = df['label_10'].values

    text_columns = ['description', 'location', 'name']
    df['combined_text'] = df[text_columns].fillna('').astype(str).apply(
        lambda row: ' '.join([s.strip() for s in row if s.strip() != '']), axis=1
    )
    texts = df['combined_text'].tolist()

    base_social = [
        'followers_count', 'friends_count', 'statuses_count', 'favourites_count',
        'listed_count', 'verified', 'default_profile', 'protected',
        'geo_enabled', 'profile_use_background_image', 'default_profile_image',
        'has_extended_profile', 'follow_request_sent', 'is_translation_enabled',
        'contributors_enabled', 'is_translator', 'profile_background_tile'
    ]

    behavior_cols = base_social + ['original_sequence_size', 'compression_ratio', 'GLTR_bert_prob', 'GLTR_gpt2_prob', 'FDGPT_probability', 'FDGPT_criterion', 'FDGPT_tokens']
    behavior_cols = [col for col in behavior_cols if col in df.columns]

    print(f"✅ 使用的行为特征数量: {len(behavior_cols)}")

    X_behavior = df[behavior_cols].copy()
    bool_cols = X_behavior.select_dtypes(include='bool').columns
    X_behavior[bool_cols] = X_behavior[bool_cols].astype(int)
    X_behavior = X_behavior.fillna(0).values
    scaler = StandardScaler()
    X_behavior = scaler.fit_transform(X_behavior)

    X_temp_texts, X_text_test, X_temp_behav, X_behav_test, y_temp, y_test = train_test_split(
        texts, X_behavior, y, test_size=0.2, random_state=42, stratify=y
    )
    X_text_train, X_text_val, X_behav_train, X_behav_val, y_train, y_val = train_test_split(
        X_temp_texts, X_temp_behav, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"训练集大小: {len(y_train)} | 验证集: {len(y_val)} | 测试集: {len(y_test)}")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 256
    BATCH_SIZE = 8
    EPOCHS = 10
    PATIENCE = 3

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SocialBotDataset(X_text_train, X_behav_train, y_train, tokenizer, MAX_LEN)
    val_dataset = SocialBotDataset(X_text_val, X_behav_val, y_val, tokenizer, MAX_LEN)
    test_dataset = SocialBotDataset(X_text_test, X_behav_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = GPT2SocialBotDetector(
        behavior_dim=len(behavior_cols),
    ).to(DEVICE)

    gpt2_params = list(model.gpt2.parameters())
    mlp_params = list(model.mlp_behavior.parameters()) + list(model.fuse.parameters())
    optimizer = AdamW([
        {'params': gpt2_params, 'lr': 1e-5},
        {'params': mlp_params, 'lr': 2e-4}
    ])

    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            behavior = batch['behavior'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, behavior)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                behavior = batch['behavior'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                logits = model(input_ids, attention_mask, behavior)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print("\n" + "="*60)
    print("🔍 加载最佳模型进行最终测试评估...")
    model.load_state_dict(torch.load(model_save_path))

    acc, prc, rec, f1, roc, bep, cm = evaluate_metrics(model, test_loader, DEVICE)

    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Precision: {prc:.4f}")
    print(f"✅ Recall   : {rec:.4f}")
    print(f"✅ F1-score : {f1:.4f}")
    print(f"✅ ROC-AUC  : {roc:.4f}")
    print(f"✅ Balanced Accuracy (BEP): {bep:.4f}")
    print("✅ Confusion Matrix:")
    print(cm)

    print("\n📊 详细分类报告:")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            behavior = batch['behavior'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            logits = model(input_ids, attention_mask, behavior)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=['human', 'bot']))

if __name__ == "__main__":
    input_file = "feature_user_behavior_GLTR_FDGPT_label_results.csv"
    model_save_path = "../models/best_gpt2_socialbot_model.pt"
    train_model(input_file, model_save_path)

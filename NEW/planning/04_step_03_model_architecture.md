# Step 3: LightCardiacNet Model Architecture

## Objective
Implement the LightCardiacNet Bi-GRU ensemble architecture.

## Architecture Overview

```
Input (seq_len, 39) ──► Bi-GRU ──► Attention ──► Dense ──► Output (2)
```

## Tasks

### 3.1 Implement Attention Layer
```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, rnn_output):
        # rnn_output: (batch, seq_len, hidden*2)
        weights = F.softmax(self.attention(rnn_output), dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context, weights
```

### 3.2 Implement Single Bi-GRU Network
```python
class BiGRUNetwork(nn.Module):
    def __init__(self, input_size=39, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = AttentionLayer(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        rnn_out, _ = self.bigru(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context)
        return logits, attn_weights
```

### 3.3 Implement Ensemble Model
```python
class LightCardiacNet(nn.Module):
    def __init__(self, input_size=39, hidden_size=128):
        super().__init__()
        self.network1 = BiGRUNetwork(input_size, hidden_size)
        self.network2 = BiGRUNetwork(input_size, hidden_size)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        logits1, attn1 = self.network1(x)
        logits2, attn2 = self.network2(x)
        
        w = torch.sigmoid(self.ensemble_weight)
        ensemble_logits = w * logits1 + (1 - w) * logits2
        
        return ensemble_logits, (attn1, attn2)
```

### 3.4 Create Dataset Class
```python
class HeartSoundDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
    
    def __getitem__(self, idx):
        patient_id = self.df.iloc[idx]['Patient ID']
        murmur_label = self.df.iloc[idx]['Murmur_Label']
        
        # Load all valve files for patient
        valve_files = glob(f"{self.data_dir}/{patient_id}_*.wav")
        
        features_list = []
        for f in valve_files:
            features = extract_features(f)
            features_list.append(features)
        
        return features_list, murmur_label
```

## Deliverables
- [ ] `model/attention.py` - Attention mechanism
- [ ] `model/lightcardiacnet.py` - Main model
- [ ] `model/dataset.py` - Data loading

## Estimated Time
~3-4 hours

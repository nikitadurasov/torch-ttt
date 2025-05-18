"""
Test-Time Training with BERT
==============================================

Discover how Masked Test-Time Training can improve model performance under data corruption.

In this tutorial, we use a pretrained BERT model fine-tuned on the Amazon Reviews dataset. During inference, we show how simple text corruptions can significantly degrade the model's performance and how the `MaskedTTT <https://torch-ttt.github.io/_autosummary/torch_ttt.engine.masked_ttt_engine.MaskedTTTEngine.html>`_ engine can natively enhance performance for BERT-based models.
"""

# %%
print("Uncomment the line below to install torch-ttt if you're running this in Colab")
# !pip install torch-ttt
# %%
# Model Fine-Tuning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this first part, we will use a widely adopted pretrained BERT model and fine-tune it on a Amazon Reviews classification task. Below, we walk through data loading and preparation, loading the pretrained BERT model, and its fine-tuning.
#
# Data Preparation
# ^^^^^^^^^^^^^^^^^^^^
# We will work with the Amazon Reviews dataset, which contains customer reviews for a wide range of products.
# To simulate distribution shift, we will use both a clean version of the dataset and a corrupted version that includes noise in the form of typos and text perturbations.
# For this tutorial, we'll focus on a single corruption type: character-level typos.
#
# Let's start by downloading the training and validation splits of our Amazon Reviews dataset.


# %%
import os

# Download Amazon Review "electronics" category training set
os.system("gdown 1mpswNa37wFhcGd-U07nMUFeF0_X4AEMi")

# Download Amazon Review "electronics" category validation set
os.system("gdown 1UbhmtFn7GEMmcoKdb2lbpf54yZV9mLoF")

# %%
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm
from copy import deepcopy

# sphinx_gallery_thumbnail_path = '_static/images/examples/bert_horizonal.png'



# %%%
# Loading Amazon Reviews
# ^^^^^^^^^^^^^^^^^^^^
# Each review consists of the review text and a score (from 0 to 4), ranging from most negative to most positive. As we will see below, we treat this as a text classification task, where each review is classified into one of these five classes.

# %%
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import Subset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AmazonReviewDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        # Load the list of (text, label, input_ids)
        self.data = torch.load(path)  # or use json.load(...) if it's a JSON file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, input_ids = self.data[idx]
        label = torch.tensor(label)
        input_ids = torch.tensor(input_ids)
        return text, label, input_ids

train_data = AmazonReviewDataset("train_amazon_review_electronics.pt")
val_data = AmazonReviewDataset("test_amazon_review_electronics.pt")

# Review Examples
review_sample = train_data[0]
print("*"*20 + " Review #1 " + "*"*20)
print(f"Review text: {review_sample[0]}")
print(f"Review score: {review_sample[1]} (from 0 to 4)\n")

review_sample = train_data[10]
print("*"*20 + " Review #2 " + "*"*20)
print(f"Review text: {review_sample[0]}")
print(f"Review score: {review_sample[1]}\n")

review_sample = train_data[100]
print("*"*20 + " Review #3 " + "*"*20)
print(f"Review text: {review_sample[0]}")
print(f"Review score: {review_sample[1]}\n")

# %%%
# Loading Pretrained BERT
# ^^^^^^^^^^^^^^^^^^^^^^^
# In this subsection, we load a pretrained BERT model with a Masked Language Modeling (MLM) head. 
# We demonstrate how BERT predicts missing words by masking a token in a sentence and retrieving 
# the top predictions using the MLM head. This showcases the model's ability to understand context 
# and semantics in text, which we will later leverage in our Test-Time Training approach.

# %%
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertForMaskedLM

# Load tokenizer, config, and base BERT (no head)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #bert-base-uncased
config = BertConfig.from_pretrained("bert-base-uncased")

# Load a full BERT with pretrained MLM head
mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Extract encoder and MLM head from it
bert = mlm_model.bert
mlm_head = mlm_model.cls


# %%%
# In the example below, we mask one of the words in a sentence and use the MLM head to predict 
# the missing token. The model returns the most likely candidates to fill in the blank, illustrating 
# its contextual understanding of language.

text = "The Milky Way is a spiral [MASK]."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Get index of [MASK] token
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# Forward pass through BERT to get hidden states
with torch.no_grad():
    outputs = bert(**inputs, return_dict=True)
    hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

# Apply MLM head to get logits over vocabulary
logits = mlm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

# Get logits at [MASK] position
mask_logits = logits[0, mask_token_index, :]  # shape: (1, vocab_size)

# Get top-3 predicted tokens
top_3_tokens = torch.topk(mask_logits, k=5, dim=1).indices[0].tolist()

# Print predictions
print("\n\n" + "*"*20 + " BERT-based Token Reconstruction " + "*"*20)
print("Original:", text)
for i, token_id in enumerate(top_3_tokens):
    predicted_token = tokenizer.decode([token_id])
    filled_text = text.replace(tokenizer.mask_token, predicted_token)
    print(f"{i+1}. {filled_text}")


# %%%
# Fine-tuning BERT
# ^^^^^^^^^^^^^^^^^^^^
# We now fine-tune the pretrained BERT encoder for a multi-class sentiment classification task using the Amazon Reviews dataset. To address class imbalance, we compute class frequencies and apply a *WeightedRandomSampler* to ensure balanced training.
#
# We define a custom classification head on top of BERT, consisting of several fully connected layers, and use the [CLS] token representation for prediction. The model is trained using cross-entropy loss with class weights, and a learning rate scheduler is set up for stable optimization.

# %%
import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda"

# %%
from collections import Counter

# Class frequencies (heavily imbalanced)
labels = [train_data[i][1].item() for i in range(len(train_data))]
sorted_freq = sorted(Counter([int(x) for x in labels]).items(), key=lambda x: x[0])
for f in sorted_freq:
  print(f"Class {f[0]}: {f[1]} samples")

# %%
from torch.utils.data import WeightedRandomSampler

# Compute weights per class
class_counts = np.bincount(labels)
class_weights = 1. / class_counts
sample_weights = class_weights[labels]

# Create weighted sampler for Dataloader
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# %%
# Class sample counts
counts = torch.tensor([x[1] for x in sorted_freq], dtype=torch.float)

# Inverse frequency
inv_freq = 1.0 / counts

# Normalize (optional but recommended)
weights = inv_freq / inv_freq.sum()

# Classification head (on top of BERT)
class BertForClassification(nn.Module):
    def __init__(self, bert, mlm_head, num_classes=2):
        super().__init__()
        self.bert = bert  # the encoder you already have
        self.dropout = nn.Dropout(0.0)
        self.mlm_head = mlm_head
        self.classifier = nn.Sequential(
            nn.Linear(bert.config.hidden_size, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        labels = kwargs.get("labels", None)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Take [CLS] token
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        logits = self.classifier(x)

        mlm_embeddings = self.mlm_head(outputs.last_hidden_state)

        # Optional loss
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=weights.to(logits.device))(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Initialize classifier model
classifier = BertForClassification(bert, mlm_head, num_classes=5).to(device)
classifier = classifier.train()

# %%
# **Training parameters**: We use the standard BERT tokenizer from HuggingFace. All training parameters—such as the number of epochs, batch size, and learning rate—are listed below.

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Custom collate function for DataLoader
def collate_fn(batch):
    # print(batch[0])
    texts = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch])
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return {**tokenized, 'labels': labels}

num_workers = 32
epochs_num = 20
train_loader = DataLoader(train_data, batch_size=num_workers, num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=num_workers, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

# Optimizer
classifier.to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100,
                             num_training_steps=len(train_loader)*epochs_num)

# %%
# **Training loop**: Here we fine-tune the BERT classifier using weighted cross-entropy and the AdamW optimizer with a learning rate scheduler. After each epoch, we evaluate accuracy on the validation set to track performance.

for epoch in range(epochs_num):
    classifier.train()
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = classifier(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Validation
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy after epoch {epoch+1}: {acc:.4f}")

# %%
# **Validation Evaluation**: After fine-tuning, our BERT model achieves approximately 60% accuracy on the Amazon Reviews test set.
classifier = classifier.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = classifier(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy (FINAL): {acc:.4f}")


# %%%
# Original BERT Evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this section, we simulate real-world noise by applying text corruptions—such as character swaps, deletions, and word shuffling—to the validation set. We define a *CorruptedDataset* wrapper that dynamically applies these perturbations during data loading.
#
# We then evaluate the fine-tuned BERT classifier from the previous sections on this corrupted data to measure its robustness. As expected, even minor textual noise can lead to a noticeable drop in performance, highlighting the need for techniques like Test-Time Training.

# %%%
# Evaluating BERT on Corrupted Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We start by defining simple text corruption functions that simulate real-world noise, such as typos and word scrambling.
# These include character-level operations (e.g., dropping, swapping, or repeating characters) and word-level perturbations 
# (e.g., deleting or shuffling words). We combine these operations to create corrupted versions of clean input text,
# which will later be used to assess how well the fine-tuned BERT model performs on noisy inputs.

# %%
import random

def corrupt_characters(text, prob=0.1):
    """Randomly drop, swap, and repeat characters."""
    corrupted = []
    for word in text.split():
        chars = list(word)
        for i in range(len(chars)):
            if random.random() < prob:
                op = random.choice(['swap', 'drop', 'repeat'])
                if op == 'swap' and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif op == 'drop':
                    chars[i] = ''
                elif op == 'repeat':
                    chars[i] = chars[i] * 2
        corrupted.append(''.join(chars))
    return ' '.join(corrupted)

def corrupt_words(text, prob=0.2):
    """Randomly drop and shuffle words."""
    words = text.split()
    new_words = []

    for word in words:
        if random.random() < prob:
            op = random.choice(['delete', 'shuffle'])
            if op == 'delete':
                continue
            elif op == 'shuffle' and len(word) > 3:
                middle = list(word[1:-1])
                random.shuffle(middle)
                word = word[0] + ''.join(middle) + word[-1]
        new_words.append(word)
    return ' '.join(new_words)

def corrupt_text(text, char_prob=0.1, word_prob=0.2):
    return corrupt_characters(corrupt_words(text, word_prob), char_prob)

# %%
# **Text corruption example**: Below we show how a clean sentence is transformed by our corruption functions,
# simulating real-world noise such as typos, deletions, and character swaps.

text = "This is an example of a clean sentence"

print(f"Original text: {text}")

corrupted_text = corrupt_text(text, 0.1, 0.1)
print(f"Corrupted text: {corrupted_text}")

# %%
# Lets define a dataset wrapper that applies text corruptions on-the-fly during data loading. 
# This allows us to evaluate the model's robustness without modifying the original dataset.

class CorruptedDataset(Dataset):

    def __init__(self, dataset, char_prob=0.1, word_prob=0.1):
        self.dataset = dataset
        self.char_prob = char_prob
        self.word_prob = word_prob

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample[0]
        corrupted_text = corrupt_text(text, self.char_prob, self.word_prob)
        return (corrupted_text, sample[1], sample[2])

    def __len__(self):
        return len(self.dataset)

# %%
corrupted_val_data = CorruptedDataset(val_data, char_prob=0.1, word_prob=0.1)
corrupted_val_loader = DataLoader(corrupted_val_data, batch_size=num_workers, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

# %%
# **Model Evaluation on Corrupted Data**: We now evaluate our fine-tuned BERT model on the corrupted validation set. As shown below, the presence of perturbations significantly degrades classification performance, with a drop of around 20%.
all_preds, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(corrupted_val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = classifier(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy after epoch: {acc:.4f}")


# %%%
# TTT-Optimized BERT Evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We evaluate the BERT model enhanced with Masked Test-Time Training (MaskedTTT) on the corrupted validation set.
# The TTT engine performs an adaptation step during inference, using masked tokens and self-supervised
# MLM loss to refine the model’s final prediction. As shown below, this approach recovers some performance lost
# due to input corruptions and improves accuracy compared to the non-adapted model.

# %%%
# Evaluating Optimized BERT on Corrupted Data
# ^^^^^^^^^^^^^^^^^^^^


# %%
from torch_ttt.engine.masked_ttt_engine import MaskedTTTEngine

# %%
engine = MaskedTTTEngine(
    model=deepcopy(classifier),
    mask_token_id=tokenizer.mask_token_id,
    features_layer_name="mlm_head.predictions",
    mask_prob=0.1,
    skip_tokens=[tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
).eval()

engine.optimization_parameters["num_steps"] = 1
engine.optimization_parameters["lr"] = 3e-5

# %%
engine.eval()
all_preds, all_labels = [], []
for i, batch in enumerate(tqdm(corrupted_val_loader, total=len(corrupted_val_loader))):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs, ttt_loss = engine(batch)
    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
    labels = batch['labels'].cpu().numpy()
    all_preds.extend(preds)
    all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy after epoch: {acc:.4f}")
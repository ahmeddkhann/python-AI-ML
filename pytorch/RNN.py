"""
train_rnn.py  -  Train an RNN that answers football questions (PyTorch)

Idea: every fact (e.g. "who has the most Champions League titles") has several
differently-worded questions. The RNN reads a question word by word and
predicts WHICH FACT it is asking about (a classification problem). The answer
text is then looked up from that class. Because paraphrases share the same
class, slightly reworded questions still get the same answer.

Run:  python train_rnn.py
"""
import csv
import random
import re
import unicodedata
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------ settings
DATA_PATH = "football_qa_dataset.csv"
MODEL_PATH = "football_rnn.pt"

EMBED_DIM = 50          # each word -> vector of 50 numbers
HIDDEN_DIM = 128        # size of the RNN memory
NUM_LAYERS = 1
BIDIRECTIONAL = True    # read the sentence left->right AND right->left
DROPOUT = 0.3
WORD_DROPOUT = 0.10     # randomly hide 10% of words while training (robustness)

EPOCHS = 120
BATCH_SIZE = 32
LR = 0.003
SEED = 42
VAL_PER_INTENT = 1      # hold out 1 paraphrase of every fact for evaluation (0 = train on everything)

PAD, UNK = "<PAD>", "<UNK>"   # special tokens
PAD_IDX, UNK_IDX = 0, 1


# ------------------------------------------------------- 1. text preprocessing
def clean_text(text: str) -> str:
    """lowercase -> strip accents -> remove punctuation -> collapse spaces"""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # anything that is not a letter/digit/space -> space
    text = re.sub(r"\s+", " ", text).strip()   # many spaces -> one space
    return text


def tokenize(text: str):
    return clean_text(text).split()


# ------------------------------------------------------------ 2. vocabulary
def build_vocab(questions):
    word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
    for q in questions:
        for w in tokenize(q):
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    return word2idx


def encode(text, word2idx):
    ids = [word2idx.get(w, UNK_IDX) for w in tokenize(text)]   # unknown word -> <UNK>
    return ids if ids else [UNK_IDX]


# --------------------------------------------------------------- 3. dataset
class QADataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples          # list of (list_of_ids, label)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, label = self.samples[i]
        if self.train and len(ids) > 3:  # word dropout: replace some words with <UNK>
            ids = [UNK_IDX if random.random() < WORD_DROPOUT else t for t in ids]
        return torch.tensor(ids), label


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)  # same length in a batch
    return padded, lengths, torch.tensor(labels)


# ------------------------------------------------------------------ 4. model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))                       # (B, T, 50)
        # pack so the RNN ignores <PAD> positions and stops at the real last word
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)                                   # h_n: (layers*dirs, B, H)
        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)                # forward + backward final states
        else:
            h = h_n[-1]
        return self.fc(self.dropout(h))                             # (B, num_classes)


# -------------------------------------------------------------- 5. evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


# ----------------------------------------------------------------- 6. main
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- load csv
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print("Questions loaded:", len(rows))

    # ---- one class per fact (intent_id); remember its answer + category
    intent_ids = sorted({int(r["intent_id"]) for r in rows})
    intent2label = {iid: i for i, iid in enumerate(intent_ids)}
    label2answer = [None] * len(intent_ids)
    by_intent = defaultdict(list)
    for r in rows:
        lab = intent2label[int(r["intent_id"])]
        label2answer[lab] = r["answer"]
        by_intent[lab].append(r)
    print("Number of classes (distinct facts):", len(intent_ids))

    # ---- split: hold out reworded questions of every fact for validation
    train_rows, val_rows = [], []
    for lab, items in by_intent.items():
        random.shuffle(items)
        k = VAL_PER_INTENT if len(items) > VAL_PER_INTENT else 0
        val_rows += [(r, lab) for r in items[:k]]
        train_rows += [(r, lab) for r in items[k:]]
    print(f"Train: {len(train_rows)}  Validation: {len(val_rows)}")

    # ---- vocabulary from TRAIN questions only (val words unseen in train -> <UNK>)
    word2idx = build_vocab([r["question"] for r, _ in train_rows])
    print("Vocabulary size:", len(word2idx))

    train_ds = QADataset([(encode(r["question"], word2idx), lab) for r, lab in train_rows], train=True)
    val_ds = QADataset([(encode(r["question"], word2idx), lab) for r, lab in val_rows])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)

    # ---- model / loss / optimiser
    model = RNNClassifier(len(word2idx), EMBED_DIM, HIDDEN_DIM, len(intent_ids),
                          NUM_LAYERS, DROPOUT, BIDIRECTIONAL).to(device)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops exploding gradients in RNNs
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        if epoch == 1 or epoch % 10 == 0:
            msg = f"Epoch {epoch:3d} | train loss {loss_sum/total:.4f} acc {correct/total:.3f}"
            if len(val_ds):
                v_loss, v_acc = evaluate(model, val_loader, device)
                msg += f" | val loss {v_loss:.4f} acc {v_acc:.3f}"
            print(msg)

    # ---- final evaluation, per category
    if len(val_ds):
        v_loss, v_acc = evaluate(model, val_loader, device)
        print(f"\nFINAL validation accuracy on unseen rewordings: {v_acc*100:.1f}%")
        model.eval()
        per_cat = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for r, lab in val_rows:
                ids = torch.tensor([encode(r["question"], word2idx)])
                pred = model(ids.to(device), torch.tensor([ids.shape[1]])).argmax(1).item()
                per_cat[r["category"]][0] += int(pred == lab)
                per_cat[r["category"]][1] += 1
        print("Accuracy per category:")
        for cat, (c, t) in sorted(per_cat.items()):
            print(f"  {cat:<10} {c}/{t}  ({100*c/t:.1f}%)")

    # ---- save everything the test script needs
    torch.save({
        "model_state": model.state_dict(),
        "word2idx": word2idx,
        "label2answer": label2answer,
        "config": dict(vocab_size=len(word2idx), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                       num_classes=len(intent_ids), num_layers=NUM_LAYERS,
                       dropout=DROPOUT, bidirectional=BIDIRECTIONAL),
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
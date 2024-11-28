import torch
import torch.nn as nn
from torchcrf import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast
import os


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 加载数据函数
def load_data(texts_file, ner_tags_file, ix_to_tag_file):
    texts = []
    ner_tags = []
    ix_to_tag = {}

    # 加载文本数据
    with open(texts_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().split()  # 按空格分割单词
            texts.append(sentence)

    # 加载标签数据
    with open(ner_tags_file, 'r', encoding='utf-8') as f:
        for line in f:
            tags = list(map(int, line.strip().split()))  # 转换为整数列表
            ner_tags.append(tags)

    # 加载标签到索引的映射
    with open(ix_to_tag_file, 'r', encoding='utf-8') as f:
        ix_to_tag = ast.literal_eval(f.read().strip())

    return texts, ner_tags, ix_to_tag


# 处理数据：将文本和标签转换为索引
def prepare_data(texts, ner_tags, max_seq_len):
    sentence_indices = []
    tag_indices = []

    for sentence, tags in zip(texts, ner_tags):
        sentence_idx = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence]
        tag_idx = tags
        sentence_indices.append(sentence_idx)
        tag_indices.append(tag_idx)

    return sentence_indices, tag_indices


# 定义BiLSTM-CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, dropout_rate=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=dropout_rate)
        self.hidden2tag = nn.Linear(HIDDEN_DIM * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentence, tags):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return -self.crf(emissions, tags)

    def predict(self, sentence):
        with torch.no_grad():
            embeds = self.embedding(sentence)
            lstm_out, _ = self.lstm(embeds)
            emissions = self.hidden2tag(lstm_out)
            return self.crf.decode(emissions)


# 自定义Dataset类
class NERDataset(Dataset):
    def __init__(self, sentences, labels, max_seq_len):
        self.sentences = sentences
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.labels[idx]

        # 对句子和标签进行padding或截断
        if len(sentence) > self.max_seq_len:
            sentence = sentence[:self.max_seq_len]
            tags = tags[:self.max_seq_len]
        else:
            sentence = sentence + [word_to_ix['<UNK>']] * (self.max_seq_len - len(sentence))
            tags = tags + [tag_to_ix[0]] * (self.max_seq_len - len(tags))

        return torch.tensor(sentence, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


# 打印数据集信息
def nums_print(li):
    total = 0
    for label, label_name in [(1, "人名(PER)"), (3, "地点(LOC)"), (5, "组织(ORG)")]:
        num = sum([1 for line in li for i in line if i == label])
        total += num
        print(f"{label_name}: {num}  ||  ", end="")
    print(f"总数: {total}")


def evaluate_on_val(model, val_sentences, val_labels):
    model.eval()  # 设置模型为评估模式
    true_labels = []
    predictions = []

    with torch.no_grad():
        for sentence, label in zip(val_sentences, val_labels):
            sentence_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(device)
            predicted_tags = model.predict(sentence_tensor)

            true_labels.extend(label)
            predictions.extend(predicted_tags[0])

    total = 0
    for i in true_labels:
        if i != 0:
            total += 1
    accuracy = sum([1 for true, pred in zip(true_labels, predictions) if pred == true != 0]) / total

    # 生成分类报告
    report = classification_report(true_labels, predictions, target_names=list(ix_to_tag.values()),
                                   labels=list(ix_to_tag.keys()), output_dict=True, zero_division=1)

    f1 = report["macro avg"]["f1-score"]

    return accuracy, f1


# 定义批次的padding函数
def pad_batch(batch):
    sentences, tags = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix['<UNK>'])
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag_to_ix[0])
    return padded_sentences, padded_tags


# 训练模型
def train_model(model, train_dataloader, val_sentences, val_labels, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    pre = 0

    for epoch in range(epochs):
        total_loss = 0
        for sentence_tensor, tag_tensor in train_dataloader:
            model.zero_grad()
            sentence_tensor, tag_tensor = sentence_tensor.to(device), tag_tensor.to(device)
            loss = model(sentence_tensor, tag_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 在验证集上进行评估
        accuracy, f1 = evaluate_on_val(model, val_sentences, val_labels)
        print(f" ========   Epoch {epoch + 1}, Loss: {total_loss:.4f}", end="   ============ 在验证集上   ")
        print(f"accuracy, f1 : {accuracy:.4f}, {f1:.4f}")
        if 0.4 * accuracy + 0.6 * f1 >= pre:
            torch.save(model.state_dict(), "model.pth")
            pre = f1
            print("saved model")
        model.train()  # 切换回训练模式，以便进行下一轮训练

# 评估模型
def evaluate_model(model, test_sentences, test_labels):
    model.eval()  # 设置模型为评估模式
    true_labels = []
    predictions = []
    bio_predictions = []

    with torch.no_grad():
        for sentence, label in zip(test_sentences, test_labels):
            sentence_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(device)
            predicted_tags = model.predict(sentence_tensor)

            true_labels.extend(label)
            predictions.extend(predicted_tags[0])

            # 将预测结果转换为BIO格式
            sentence_bio = [ix_to_tag[tag] for tag in predicted_tags[0]]
            bio_predictions.append(sentence_bio)

    total = 0
    for i in true_labels:
        if i != 0:
            total += 1
    accuracy = sum([1 for true, pred in zip(true_labels, predictions) if pred == true != 0]) / total

    # 生成分类报告
    report = classification_report(true_labels, predictions, target_names=list(ix_to_tag.values()),
                                   labels=list(ix_to_tag.keys()), output_dict=True, zero_division=1)

    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]
    f1 = report["macro avg"]["f1-score"]

    print("\n>>>>>>>>>>>>>>  评价模型  <<<<<<<<<<<<<<<")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # 打印前10个句子的预测结果
    print("\n>>>>>>>>>>>>>>  打印测试集前10个句子的预测结果  <<<<<<<<<<<<<<<")
    print("Predictions in BIO format:")
    for sentence, bio_pred in zip(test_sentences[:10], bio_predictions[:10]):
        sentence_words = [ix_to_word.get(idx, "<UNK>") for idx in sentence]
        print(f"Sentence: {' '.join(sentence_words)}")
        print(f"Predictions (BIO): {' '.join(bio_pred)}\n")


# 预测单个句子
def predict_sentence(model, sentence):
    model.eval()
    sentence_tensor = torch.tensor([word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence],
                                   dtype=torch.long).unsqueeze(0).to(device)
    predicted_tags = model.predict(sentence_tensor)

    return [(word, ix_to_tag[tag]) for word, tag in zip(sentence, predicted_tags[0])]


"""

      载入数据文件

"""

# 加载数据
texts, ner_tags, ix_to_tag = load_data('texts.txt', 'tags.txt', "id_to_label_msra_ner.txt")     #

# 创建词汇表和标签映射
word_to_ix = {word: i for i, word in enumerate(sorted(set(word for sentence in texts for word in sentence)))}
tag_to_ix = {tag: i for i, tag in enumerate(sorted(set(tag for tags in ner_tags for tag in tags)))}
word_to_ix['<UNK>'] = len(word_to_ix)  # 添加 <UNK> 标记
ix_to_word = {i: word for word, i in word_to_ix.items()}

"""

        主程序

"""
# 超参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 48
MAX_SEQ_LEN = 128
DROPOUT_RATE = 0.5
EPOCH = 20

# 准备数据
sentence_indices, label_indices = prepare_data(texts, ner_tags, MAX_SEQ_LEN)

# 划分训练集和测试集
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentence_indices, label_indices, test_size=0.2, random_state=42)
# 划分训练集和验证集
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_sentences, train_labels, test_size=0.1, random_state=42, shuffle=False)  # 10% 作为验证集


print("Train Data Info")
nums_print(train_labels)
print("Test Data Info")
nums_print(test_labels)

# 创建DataLoader
train_dataset = NERDataset(train_sentences, train_labels, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)

# 创建模型并移动到GPU
model = BiLSTM_CRF(len(word_to_ix), len(tag_to_ix), dropout_rate=DROPOUT_RATE).to(device)


model_file = "model.pth"
if os.path.exists(model_file):
    use_existing_model = input(
        f"Model file found. Do you want to load the existing model from {model_file}? (y/n): ").lower()
    if use_existing_model == "y":
        model.load_state_dict(torch.load(model_file))
        model.eval()
        print("Model loaded successfully.")
    else:
        os.remove(model_file)
        print("Model deleted")
        # Train the model
        train_model(model, train_dataloader, val_sentences, val_labels, epochs=EPOCH)
        model.eval()
else:
    # 训练模型
    train_model(model, train_dataloader, val_sentences, val_labels, epochs=EPOCH)
    model.eval()


# 评估模型
evaluate_model(model, test_sentences, test_labels)

# 单句测试
sentence = ["你", "好", "我", "是", "王", "晓", "明", "北", "京", "是", "中", "国", "的", "首", "都"]
predictions = predict_sentence(model, sentence)
print("\n单句测试:")
print(predictions)

from datasets import load_dataset

# 加载 数据集的前 100 条数据
dataset = load_dataset("msra_ner", split='train[:10000]')  #, split='train[:1000]'

# 提取文本和标签
texts = [item['tokens'] for item in dataset]
ner_tags = [item['ner_tags'] for item in dataset]

# 将 texts 写入文本文件
with open('texts.txt', 'w', encoding='utf-8') as f:
    for sentence in texts:
        f.write(' '.join(sentence) + '\n')

# 将 ner_tags 写入文本文件
with open('tags.txt', 'w', encoding='utf-8') as f:
    for tags in ner_tags:
        f.write(' '.join(map(str, tags)) + '\n')

print("文本和标签已成功写入文件.")

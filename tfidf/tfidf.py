import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
import math
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from collections import Counter
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA


class TFIDF(object):

    def __init__(self, corpus):
        """
        初始化
        self.vob:词汇个数统计，dict格式
        self.word_id:词汇编码id，dict格式
        :param corpus:输入的语料
        """
        self.smooth_idf = 0.01
        self.word_id = {}
        self.vob = {}
        self.corpus = corpus

    def get_vob_fre(self):
        """
        计算文本频率term frequency，词id
        :return: 修改self.vob也就是修改词频统计字典
        """
        # 统计各词出现个数
        id = 0
        for single_corpus in self.corpus:
            if isinstance(single_corpus, list):
                pass
            if isinstance(single_corpus, str):
                single_corpus = single_corpus.strip("\n").split(" ")
            for word in single_corpus:
                if word not in self.vob:
                    self.vob[word] = 1
                    self.word_id[word] = id
                    id += 1
                else:
                    self.vob[word] += 1

        # 生成矩阵
        X = np.zeros((len(self.corpus), len(self.vob)))
        for i in range(len(self.corpus)):
            if isinstance(self.corpus[i], str):
                single_corpus = self.corpus[i].strip("\n").split(" ")
            else:
                single_corpus = self.corpus[i]
            for j in range(len(single_corpus)):
                feature = single_corpus[j]
                feature_id = self.word_id[feature]
                X[i, feature_id] = self.vob[feature]
        return X.astype(int)  # 需要转化成int

    def get_tf_idf(self):
        """
        计算idf并生成最后的TFIDF矩阵
        :return:
        """
        X = self.get_vob_fre()
        n_samples, n_features = X.shape
        df = []
        for i in range(n_features):
            # 这里是统计每个特征的非0的数量，也就是逆文档频率指数的分式中的分母，是为了计算idf
            df.append(n_samples - np.bincount(X[:, i])[0])
        df = np.array(df)
        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_samples += int(self.smooth_idf)
        idf = np.log(n_samples / df) + 1
        matrix = X*idf/len(self.vob)
        transpose = np.transpose(matrix)
        return transpose


def mypca(w, dimension):
    pca = PCA(n_components=dimension)  # 初始化PCA
    X = pca.fit_transform(w)  # 返回降维后的数据
    return X


# 初始化vocab
with open("word_freq.txt", encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab = set(vocab)

'''
dataset = pd.read_csv("IMDB Dataset.csv")
#数据清洗&文本预处理
corpus = []
sentiment = []
# fword = open("word.txt", 'w', encoding='utf-8')
for i in range(50000):
    pattern = re.compile(r'<[^>]+>', re.S)
    review = pattern.sub('', dataset['review'][i])
    review = re.sub('[^a-zA-z]', ' ', review)
    review = review.lower()
    review = review.split()
    # for word in review:
        # fword.write(word + "\n")
    review = [word for word in review if word in vocab]
    sen = dataset['sentiment'][i]
    if len(review)==0:
        if sen =='positive':
            review.append('wonderful')
        else:
            review.append('boring')
    review = ' '.join(review)
    corpus.append(review)
    if sen == 'positive':
        sentiment.append(1)
    else:
        sentiment.append(0)
# fword.close()
dataframe = pd.DataFrame({'sentiment': sentiment, 'review': corpus})
dataframe.to_csv("clean.csv")

'''
# 载入预处理后的数据
train_data = pd.read_csv("clean.csv", index_col=0)

'''
# 载入词表
f = open('word.txt', 'r')
sourceInLines = f.readlines()  #按行读出文件内容
f.close()
words = []
for line in sourceInLines:
    temp1 = line.strip('\n')       #去掉每行最后的换行符'\n'
    words.append(temp1)

# 去除高低频词,去除后的词存于“word_freq.txt”中
Min = 1000
Max = 25000
with open("word_freq.txt", 'w', encoding='utf-8') as fout:
    for word, freq in Counter(words).most_common():
        if freq > Min and freq < Max:
            fout.write(word+"\n")

'''

# CNN输入数据处理：将句子转换为编号列表
word2idx = {i: index for index, i in enumerate(vocab)}
idx2word = {index: i for index, i in enumerate(vocab)}
vocab_size = len(vocab)

# 计算review长度，好选定截断review的长度，这里选择286
comments_len = []
corpus = train_data['review']
i = 0
for review in corpus:
    review = review.split()
    comments_len.append(len(review))
    i = i+1
train_data["comments_len"] = comments_len
print(train_data["comments_len"].describe(percentiles=[.5, .95]))

# 对句子用编号表示，对句子进行截断/补全
pad_id = word2idx["two"]# 填充使用”two“(无情感词)
sequence_length = 286# 句子固定长度

def tokenizer():
    inputs = []
    sentence_char = [i.split() for i in train_data["review"]]
    for index, i in enumerate(sentence_char):
        # 转换为编号列表
        temp = [word2idx.get(j, pad_id) for j in i]
        if len(i) < sequence_length:
            # 补全
            for _ in range(sequence_length-len(i)):
                temp.append(pad_id)
        else:
            # 截断
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs


data_input = tokenizer()

# 参数定义
device = "cpu"
Embedding_size = 250
Batch_Size = 64
Kernel = 3
Filter_num = 10# 卷积核的数量。
Epoch = 30
Dropout = 0.5
Learning_rate = 1e-3
num_classs = 2# 2分类问题

corpus = train_data['review']
# tfidf作为向量
tfidf = TFIDF(corpus)
w2v = tfidf.get_tf_idf()
w2v = mypca(w2v, 250)

class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


# 划分数据集
TextCNNDataSet = TextCNNDataSet(data_input, list(train_data["sentiment"]))
train_dataset = torch.utils.data.Subset(TextCNNDataSet, range(0, 30000))
val_dataset = torch.utils.data.Subset(TextCNNDataSet, range(30000, 40000))
test_dataset = torch.utils.data.Subset(TextCNNDataSet, range(40000, 50000))

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)
ValDataLoader = Data.DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True)



def word2vec(x):
    # x是编号的形式
    x2v = np.ones((len(x), x.shape[1], Embedding_size))
    for i in range(len(x)):
        temp = []
        for j in x[i]:
            w = idx2word[j.item()]
            k = tfidf.word_id[w]
            temp.append(w2v[k])
        x2v[i] = temp
    return torch.tensor(x2v).to(torch.float32)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        out_channel = Filter_num # 可以等价为通道的解释。
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),# 卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.ReLU(),
                    nn.MaxPool2d((sequence_length-1, 1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, num_classs)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = word2vec(X)
        embedding_X = embedding_X.unsqueeze(1)
        conved = self.conv(embedding_X)
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        # 2分类问题，往往使用softmax，表示概率。
        return F.log_softmax(output)


model = TextCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)

def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def train():
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = F.nll_loss(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc

def validation():
    """
    模型验证
    :param model: 使用的模型
    :return: 返回当前训练的模型在验证集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in ValDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()


# 训练
model_train_acc, model_test_acc = [], []
print("开始训练...")
for epoch in range(Epoch):
    train_acc = train()
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    if (epoch+1) % 10 == 0:
        val_acc = validation()
        print("epoch = {}, 验证集准确率={}".format(epoch + 1, val_acc))
    model_train_acc.append(train_acc)

torch.save(model, 'save.pt')
checkpoint = {"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": Epoch}
path_checkpoint = "./checkpoint_{}_epoch.pkl".format(Epoch)
torch.save(checkpoint, path_checkpoint)

def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    avg_pre = []
    avg_recall = []
    avg_f1 = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            y_pred = torch.max(pred, dim=1)[1]
            acc = binary_acc(y_pred, y_batch)
            pre = sklearn.metrics.precision_score(y_batch, y_pred)
            recall = sklearn.metrics.recall_score(y_batch, y_pred)
            f1 = sklearn.metrics.f1_score(y_batch, y_pred)
            avg_acc.append(acc)
            avg_pre.append(pre)
            avg_recall.append(recall)
            avg_f1.append(f1)
    eval_acc = np.array(avg_acc).mean()
    eval_pre = np.array(avg_pre).mean()
    eval_recall = np.array(avg_recall).mean()
    eval_f1 = np.array(avg_f1).mean()
    print("测试集准确率={}".format(eval_acc))
    print("测试集精确率={}".format(eval_pre))
    print("测试集召回率={}".format(eval_recall))
    print("测试集F1-Score={}".format(eval_f1))


evaluate()


# 展示训练过程中准确率变化
plt.plot(model_train_acc)
plt.ylim(ymin=0.5, ymax=1)
plt.title("The accuracy of textCNN model")
plt.show()




















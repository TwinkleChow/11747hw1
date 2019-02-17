from collections import defaultdict
import time
import random
import torch
import numpy as np


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags,weights):
        super(CNNclass, self).__init__()

        """ layers """
        # self.embedding = torch.nn.Embedding(nwords, emb_size)
        self.embedding = torch.nn.Embedding.from_pretrained(weights,freeze=False)
        # uniform initialization
        # torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d_list=torch.nn.ModuleList([torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size[i],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True) for i in range(len(window_size))])
        
        self.relu = torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(0)
        self.projection_layer = torch.nn.Linear(in_features=num_filters*len(window_size), out_features=ntags, bias=True)
        # Initializing the projection layer
        # torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # nwords x emb_size
        emb = emb.permute(0, 2, 1)     # 1 x emb_size x nwords
        h = []
        for i,l in enumerate(self.conv_1d_list):
            hi = l(emb)
            hi = hi.max(dim=2)[0]
            hi = self.relu(hi)
            h.append(hi)
        # Do max pooling
        # h1 = h1.max(dim=2)[0]                         # 1 x num_filters
        # h = self.relu(h)
        
        # concatenate
        h = torch.cat(h,dim=1)
        h= self.dropout(h)
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        return out


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

def read_testset(filename):
    with open(filename, "r") as f:
        test =[]
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            eg = []
            for w in words.split(" "):
                if w not in w2i:
                    eg.append(0)
                else:
                    eg.append(w2i[w])
            test.append(eg)
        return test

# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
nwords = len(w2i)
ntags = len(t2i)

test = list(read_testset("topicclass/topicclass_test.txt"))

i2t = {v:k for (k,v) in t2i.iteritems()}

# load pretrained word2vec
weights = np.loadtxt('emb.txt')
weights = torch.FloatTensor(weights.tolist())
# Define the model
EMB_SIZE = 300
WIN_SIZE = [1,2,3,4,5,6,7]
FILTER_SIZE = 100
batch_size = 256
num_iter = int(len(train)/float(batch_size))
# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags,weights)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

last_val_acc = 0.0

for epoch in range(10):
    # Perform training
    random.shuffle(train)
    start = time.time()
    train_loss = 0.0
    train_correct = 0.0
    for i in range(num_iter):
        x = train[i*batch_size:(i+1)*batch_size]
        max_len = max([len(words) for words,tag in x])
        words = [x[j][0] for j in range(len(x))]
        tags = [x[j][1] for j in range(len(x))]
        # print 'i:',i

        # pad words
        max_len = max(len(j) for j in words)
        words = [j+[0]*(max_len-len(j)) for j in words]

        # for words, tag in x:
        #     # Padding (can be done in the conv layer as well)
        #     if len(words) < max_len:
        #         words += [0] * (max_len - len(words))
        words_tensor = torch.tensor(words).type(type)
        tag_tensor = torch.tensor(tags).type(type)
        scores = model(words_tensor)
        # predict = scores[0].argmax().item()
        predict = torch.argmax(scores,dim=1)

        train_correct += torch.sum(predict==tag_tensor).item()

        my_loss = criterion(scores, tag_tensor)
        train_loss += my_loss.item()
            # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("epoch %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
            epoch, train_loss / len(train), train_correct / len(train), time.time() - start))
    
    # Perform testing on validation set

    dev_correct = 0.0
    words = [i[0] for i in dev]
    tags = [i[1] for i in dev]
    max_len = max([len(i) for i in words])
    words = [j+[0]*(max_len-len(j)) for j in words]

    # for words, tag in dev:
        # Padding (can be done in the conv layer as well)
        # if len(words) < max_len:
        #     words += [0] * (max_len - len(words))
    words_tensor = torch.tensor(words).type(type)
    tag_tensor = torch.tensor(tags).type(type)
    scores = model(words_tensor)
    # predict = scores.argmax().item()
    #     if predict == tag:
    #         test_correct += 1
    predict = torch.argmax(scores,dim=1)
    dev_correct= torch.sum(predict==tag_tensor).item()
    dev_acc = dev_correct / float(len(dev))
    print("epoch %r: dev acc=%.4f" % (epoch, dev_acc))
    if dev_acc>0.83:
        # save the result
        dev_acc = float("{0:.4f}".format(dev_acc))
        with open('dev_label_'+str(dev_acc),'wb') as f:
            for i in range(len(predict)):
                f.write(i2t[predict[i].item()])
                f.write('\n')
        print 'dev result saved'
        # output label for test set
        test_correct = 0.0
        max_len = max([len(i) for i in test])
        words = [j+[0]*(max_len-len(j)) for j in test]

        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)

        predict = torch.argmax(scores,dim=1)
        with open('test_label_'+str(dev_acc),'wb') as f:
            for i in range(len(predict)):
                f.write(i2t[predict[i].item()])
                f.write('\n')
        print 'test result saved'


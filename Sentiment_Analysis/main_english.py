import torch.nn.functional as F
import random, re, nltk, string, time, torch
from model import GCN, accuracy
import numpy as np
import scipy.sparse as sp
import networkx as nx


## Retrieving dataset

f = open('./comments/assultive_english_train.txt', 'r', encoding='utf8')
corpus_train = f.read().splitlines()
random.shuffle(corpus_train)
f.close()

f = open('./comments/assultive_english_test.txt', 'r', encoding='utf8')
corpus_test = f.read().splitlines()
random.shuffle(corpus_test)
f.close()

del f

## Test
# corpus_train = random.sample(corpus_train, 10000)
# corpus_test = random.sample(corpus_test, 1000)
##


## Splitting corpus and labels

corpus_train = [corpus.split('\t') for corpus in corpus_train]
labels_train = [int(corpus[0]) for corpus in corpus_train]
corpus_train = [''.join(corpus[1:]).strip() for corpus in corpus_train]

corpus_test = [corpus.split('\t') for corpus in corpus_test]
labels_test = [int(corpus[0]) for corpus in corpus_test]
corpus_test = [''.join(corpus[1:]).strip() for corpus in corpus_test]

## Preprocessing corpus

for alphabet in list(string.ascii_lowercase):
    corpus_train = [re.sub(str('[' + alphabet + ']' + '[' + alphabet + ']' + '[' + alphabet + ']+'), alphabet, corpus) for corpus in corpus_train]
    corpus_test = [re.sub(str('[' + alphabet + ']' + '[' + alphabet + ']' + '[' + alphabet + ']+'), '', corpus) for corpus in corpus_test]
    del alphabet

corpus_train = [re.sub('[^a-zA-Z]', ' ', corpus).lower().split() for corpus in corpus_train]
corpus_test = [re.sub('[^a-zA-Z]', ' ', corpus).lower().split() for corpus in corpus_test]

stops = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')

train_corpus = []
test_corpus = []
corpus_set = []
for i in range(len(corpus_train)):
    corpus = [stemmer.stem(word) for word in corpus_train[i] if not word in stops]
    train_corpus.append(corpus)
    corpus_set.extend(corpus)

for i in range(len(corpus_test)):
    corpus = [stemmer.stem(word) for word in corpus_test[i] if not word in stops]
    test_corpus.append(corpus)
    corpus_set.extend(corpus)

corpus = train_corpus + test_corpus
labels = labels_train + labels_test
check = np.unique(corpus_set, return_counts=True)

del stemmer, stops, corpus_train, corpus_test, labels_train, labels_test

## Tokenizing corpus

start_time = time.time()
tokens_dict = []
tokens_freq = []
k = 0
for i in range(len(list(check[1]))):
    if list(check[1])[i] > 5:
        tokens_dict.append([list(check[0])[i], k])
        tokens_freq.append(list(check[1])[i])
        k += 1

tokens_dict = dict(tokens_dict)

print("tokens",'\t',time.time() - start_time)

## Onehot vectorizing

start_time = time.time()
comments_onehot = [list(range(len(tokens_dict)))]
labels_onehot = ['test']
for comment in corpus:
    onehot = list(map(tokens_dict.get, comment))
    if onehot.count(None) > 0:
        for i in range(onehot.count(None)):
            onehot.remove(None)
    if len(onehot) > 0:
        comments_onehot.append(onehot)
        labels_onehot.append(int(labels[corpus.index(comment)]))
print("onehot",'\t',time.time() - start_time)
labels = labels_onehot

## nodes_on_features

start_time = time.time()
nodes_on_features = [None] * len(tokens_freq)
for i in range(len(comments_onehot)):
    for num in comments_onehot[i]:
        if nodes_on_features[num] is None:
            nodes_on_features[num] = [i]
        else:
            nodes_on_features[num].append(i)

del check, corpus, labels_onehot

print("nodes_on_features",'\t',time.time() - start_time)

## features

start_time = time.time()
check_adj = []
for i in range(len(tokens_freq)):
    if tokens_freq[i] > 10:
        check_adj.append(i)

features = []
adj_pair = [None] * len(check_adj)
for comment in comments_onehot:
    ind_feature = [0] * len(tokens_dict)
    for num in comment:
        if num in check_adj:
            if adj_pair[check_adj.index(num)]:
                adj_pair[check_adj.index(num)].append(comments_onehot.index(comment))
            else:
                adj_pair[check_adj.index(num)] = [comments_onehot.index(comment)]
    ind_feature[num] += 1
    features.append(ind_feature)

features = np.matrix(features)

torch.save(tokens_dict, './test1/tokens_dict.sav')
# joblib.dump(tokens_dict, './test0/tokens_dict.sav')

del tokens_dict, tokens_freq, comments_onehot, check_adj

print("features", '\t', time.time() - start_time)

## Edges

start_time = time.time()
edges_list = []
for ad in adj_pair:
    if len(ad) == 2:
        edges = (ad[0], ad[1])
        edges_list.append(edges)
    else:
        adj = list(set(ad))
        if len(adj) == 2:
            edges = (adj[0], adj[1])
            edges_list.append(edges)
        else:
            for i in range(len(adj) - 1):
                edges = (adj[i], adj[i + 1])
                edges_list.append(edges)
            edges = (adj[0], adj[len(adj) - 1])
            edges_list.append(edges)

G = nx.Graph()
G.add_nodes_from(list(range(features.shape[0])))
G.add_edges_from(edges_list)

print("Edges", '\t', time.time() - start_time)

## Adjacency Matrix

start_time = time.time()
adj = nx.adj_matrix(G).astype('float32').todense()
adj = adj + np.eye(G.number_of_nodes(), dtype='float32')

rowsum = np.array(adj.sum(1))
r_inv = np.power(rowsum, -1).flatten()
r_inv[np.isinf(r_inv)] = 0.
r_mat_inv = sp.diags(r_inv)
adj = r_mat_inv.dot(adj)

print("adj", '\t', time.time() - start_time)

features = torch.FloatTensor(features)

model = GCN(num_feat= features.shape[1],
            num_hidden1= int(2 * features.shape[1] / 3),
            num_hidden2= 16,
            num_class=len(set(labels))
            # dropout=0.01
            )

optimizer = torch.optim.Adam(model.parameters(), lr = 0.02, weight_decay= 5e-4)
labels = torch.tensor(labels[1:])
adj = torch.from_numpy(adj).to_sparse()
split_index = int(19 * len(labels) / 20)

torch.save(features, './test1/features.sav')
torch.save(nodes_on_features, './test1/nodes_on_features.sav')
torch.save(adj, './test1/adj.sav')

## GPU

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# features = features.to(torch.device("cuda"))
# adj = adj.to(torch.device("cuda"))
# labels = labels.to(torch.device("cuda"))

f = open('training_log.txt', 'w', encoding='utf8')

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[1:split_index + 1], labels[:split_index])
    acc_train = accuracy(output[1:split_index + 1], labels[:split_index])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1), '\t',
          'loss_train: {:.4f}'.format(loss_train.item()), '\t'
          'acc_train: {:.4f}'.format(acc_train.item()), '\t'
          'time: {:.4f}s'.format(time.time() - t))

    f.write('Epoch: {:04d}'.format(epoch + 1) + '\t' + 'loss_train: {:.4f}'.format(loss_train.item()) + '\t' + 'acc_train: {:.4f}'.format(acc_train.item()) + '\t' + 'time: {:.4f}s'.format(time.time() - t) + '\n')

t_total = time.time()
for epoch in range(5000):
    if (epoch + 1) % 1000 == 0:
        model_name = 'model_' + str(epoch + 1)
        path = './testset/'+ model_name
        torch.save(model, path)
    train(epoch)

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[split_index + 1:], labels[split_index:])
    acc_test = accuracy(output[split_index + 1:], labels[split_index:])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

test()

torch.save(model, './test1/model.sav')

# def test(test_comment, G, nodes_on_feature, tokens_dict, features):
#
#     features_test, adj_test = proc.test_comment(test_comment, G, nods_on_feature, tokens_dict, features)
#     adj_test = torch.from_numpy(adj_test).to_sparse()
#     model.eval()
#     output = model(features_test, adj_test)
#     prob = np.exp(output.detach())
#     prob = prob[len(prob)-1].tolist()
#     print(prob[1])
#
# test('여기에 테스트 문장을 넣어주세요!')

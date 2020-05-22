import dgl
import networkx as nx
import torch
import numpy as np
from random import seed, randrange, sample
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.pyplot as plt

## 국회의원 Sample modeling with random edges

G = dgl.DGLGraph()
G.add_nodes(300)

for i in range(G.number_of_nodes()):
    seed(i)
    # Blue party
    if i < 122:
        seq = list(np.arange(0, 122))
        seq.remove(i)
        random_range = randrange(2, 3 + round(i/20))
        edges_list = sample(seq, random_range)
        if random_range > 6:
            edges_list.append(122)
        edges_list = sorted(edges_list)
        G.add_edges(i, edges_list)
        G.add_edges(edges_list, i)
    elif i == 122:
        pass
    # Red party
    elif i in range(123, 244):
        seq = list(np.arange(123, 244))
        seq.remove(i)
        random_range = randrange(2, 3 + round((i-123)/20))
        edges_list = sample(seq, random_range)
        if random_range > 6:
            edges_list.append(244)
        edges_list = sorted(edges_list)
        G.add_edges(i, edges_list)
        G.add_edges(edges_list, i)
    elif i == 244:
        pass
    # Green party
    elif i in range(245, 282):
        seq = list(np.arange(245, 282))
        random_range = randrange(2, 3 + round((i-245)/10))
        edges_list = sample(seq, random_range)
        if random_range > 3:
            edges_list.append(282)
        edges_list = sorted(edges_list)
        G.add_edges(i, edges_list)
        G.add_edges(edges_list, i)
    elif i == 282:
        pass
    # Yellow party
    elif i in range(283, 289):
        seq = list(np.arange(283, 289))
        random_range = 2
        edges_list = sample(seq, random_range)
        edges_list.append(289)
        edges_list = sorted(edges_list)
        G.add_edges(i, edges_list)
        G.add_edges(edges_list, i)
    elif i == 289:
        pass
    # Gray party
    else:
        seq = list(np.arange(0, G.number_of_nodes()))
        random_range = 5
        edges_list = sample(seq, random_range)
        edges_list = sorted(edges_list)
        G.add_edges(i, edges_list)
        G.add_edges(edges_list, i)

seq = list(np.arange(0, G.number_of_nodes()))
main_nodes = [122, 244, 282, 289]
for i in range(len(main_nodes)):
    seq.remove(main_nodes[i])
random_range = 60
edges_list = sample(seq, random_range)
edges_src = edges_list[:30]
edges_dst = edges_list[30:60]
G.add_edges(edges_src, edges_dst)
G.add_edges(edges_dst, edges_src)

fig = plt.figure(dpi=150)
nx_G = G.to_networkx().to_undirected()
nx.draw(nx_G, pos=None, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()

labeled_A = sorted(sample(list(range(0, 122)), 29))
labeled_A.append(122)
labeled_B = sorted(sample(list(range(245, 282)), 29))
labeled_B.append(282)

labeled_nodes = torch.tensor(labeled_A + labeled_B)
labels = torch.tensor([0] * len(labeled_A) + [1] * len(labeled_B))

print('The labeled nodes to A:\t' + str(labeled_A))
print('The labeled nodes to B:\t' + str(labeled_B))
print('The ratio of labeled nodes:\t' + str(len(labeled_A + labeled_B) / G.number_of_nodes()))

## Model Set-up

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, Adjacency_matrix, feature_matrix):
        # Adding self loop
        A_hat = Adjacency_matrix + np.eye(Adjacency_matrix.shape[0])
        # Normailizing adjacency matrix
        D = np.matrix(np.diag(np.array(np.sum(A_hat, axis=0))[0]))
        # Extracting feature
        feat = np.sqrt(D) ** -1 * A_hat * np.sqrt(D) ** -1 * feature_matrix
        return self.linear(torch.from_numpy(feat.astype('float32')))

# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, Adjacency_matrix, feature_matrix):
        h = self.gcn1(Adjacency_matrix, feature_matrix)
        h = torch.relu(h)
        h = h.detach().numpy()
        h = self.gcn2(Adjacency_matrix, h)
        return h



A = nx.to_numpy_matrix(nx_G)
seed(0)
X = np.random.randint(2, size = (G.number_of_nodes(), 10)).astype('float32')
# X = np.eye(G.number_of_nodes())
net = GCN(X.shape[1],10,2)
# print('The feature of nodes')
# X

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(2000):
    logits = net(A, X)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(A.shape[0]):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=100, ax=ax)

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=100)
plt.show()

result = all_logits[len(all_logits)-1].numpy()
class_A, class_B = [], []
for v in range(result.shape[0]):
    if result[v].argmax():
        class_B.append(v)
    else:
        class_A.append(v)

print('The class_A nodes')
print(class_A)
print('\n')
print('The class_B nodes')
print(class_B)
print('\n')
print('The winning ratio of the class A is ' + str(len(class_A) / G.number_of_nodes()))

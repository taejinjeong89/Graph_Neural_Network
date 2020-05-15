import numpy as np
import random, re, torch, os, string, nltk, joblib
import networkx as nx
import scipy.sparse as sp

class processor():

    def __init__(self, main_dir):
        self.main_dir = main_dir

    def naver_ratings(self, shuffle = True):
        f = open(self.main_dir + '/naver_train.txt', 'r', encoding='utf8')
        train_comments = f.read().splitlines()
        train_comments = train_comments[1:]
        f.close()
        f = open(self.main_dir + '/naver_test.txt', 'r', encoding='utf8')
        test_comments = f.read().splitlines()
        test_comments = test_comments[1:]
        f.close()
        del f
        if shuffle:
            random.shuffle(train_comments)
            random.shuffle(test_comments)

        train_labels = [int(comment.split('\t')[2]) for comment in train_comments]
        train_comments = [comment.split('\t')[1] for comment in train_comments]
        test_labels = [int(comment.split('\t')[2]) for comment in test_comments]
        test_comments = [comment.split('\t')[1] for comment in test_comments]

        return train_comments, train_labels, test_comments, test_labels


    def comment_label(self, train_test_split = None, shuffle = True):
        f = open(self.main_dir + '/negative_comments.txt', 'r', encoding='utf8')
        negative_comments = f.read().splitlines()
        f.close()
        f = open(self.main_dir + '/positive_comments.txt', 'r', encoding='utf8')
        positive_comments = f.read().splitlines()
        f.close()
        del f

        if train_test_split is None:
            comments = negative_comments + positive_comments
            comments = [comment.split('\t') for comment in comments]
            if shuffle:
                random.shuffle(comments)
            labels = [int(comment[1]) for comment in comments]
            comments = [comment[0].strip() for comment in comments]

            # self.comments = comments
            # self.labels = labels

            return comments, labels

        else:
            if shuffle:
                random.shuffle(negative_comments)
                random.shuffle(positive_comments)
            train_neg = negative_comments[:round(len(negative_comments) * train_test_split)]
            train_pos = positive_comments[:round(len(positive_comments) * train_test_split)]
            test_neg = negative_comments[round(len(negative_comments) * train_test_split):]
            test_pos = positive_comments[round(len(positive_comments) * train_test_split):]

            comments_train = train_neg + train_pos
            comments_test = test_neg + test_pos

            comments_train = [comment.split('\t') for comment in comments_train]
            comments_test = [comment.split('\t') for comment in comments_test]

            labels_train = [int(comment[1]) for comment in comments_train]
            labels_test = [int(comment[1]) for comment in comments_test]

            comments_train = [comment[0].strip() for comment in comments_train]
            comments_test = [comment[0].strip() for comment in comments_test]

            comments = [comments_train, comments_test]
            labels = [labels_train, labels_test]

            # self.comments = comments
            # self.labels = labels

            return comments, labels

    def sentence_onehot(self, comments, labels, select_tags=['NNG', 'NNP', 'NP', 'VA', 'MAG', 'MAJ'], with_tags=True):

        ## Preprocessing Stage
        comment_set = []
        comments_proc = []

        from PyKomoran import Komoran
        komoran = Komoran("STABLE")
        komoran.set_user_dic(os.path.join('./dictionary', 'user.dic'))
        komoran.set_fw_dic(os.path.join('./dictionary', 'fwd.dic'))

        comments = [re.sub(pattern='([ㄱ-ㅎㅏ-ㅣ]+)', repl='', string=comment) for comment in comments]
        comments = [re.sub(pattern='[^\w\s]', repl='', string=comment) for comment in comments]
        comments = ['\t'.join(map(str, komoran.get_list(comment))) for comment in comments]
        if with_tags:
            for comment in comments:
                comment = re.compile(r'(?:[^\t]+)(?:{})'.format('\/' + '|\/'.join(select_tags))).findall(comment)
                comments_proc.append(comment)
                comment_set.extend(comment)
        else:
            for comment in comments:
                comment = re.compile(r'(?:[^\t]+)(?:{})'.format('\/' + '|\/'.join(select_tags))).findall(comment)
                comments_proc.append(comment)
                comment_set.extend(comment)

        ## Vocabulary matrix
        check = np.unique(comment_set, return_counts=True)

        tokens_set = []
        tokens_freq = []
        k = 0
        for i in range(len(list(check[1]))):
            if list(check[1])[i] > 1:
                tokens_set.append([list(check[0])[i], k])
                tokens_freq.append(list(check[1])[i])
                k += 1

        tokens_dict = dict(tokens_set)

        comments_onehot = []
        labels_onehot = []
        for comment in comments_proc:
            onehot = list(map(tokens_dict.get, comment))
            if onehot.count(None) > 0:
                for i in range(onehot.count(None)):
                    onehot.remove(None)
            if len(onehot) > 0:
                comments_onehot.append(onehot)
                labels_onehot.append(int(labels[comments_proc.index(comment)]))

        nodes_on_features = [None] * len(tokens_freq)
        for i in range(len(comments_onehot)):
            for num in comments_onehot[i]:
                if nodes_on_features[num] is None:
                    nodes_on_features[num] = [i]
                else:
                    nodes_on_features[num].append(i)

        # self.comments_onehot = comments_onehot
        # self.tokens_dict = tokens_dict
        # self.tokens_freq = tokens_freq
        # self.nodes_on_features = nodes_on_features

        return comments_onehot, labels_onehot, tokens_dict, tokens_freq, nodes_on_features


    def graph_generation(self, tokens_dict = None, tokens_freq = None, comments_onehot = None):

        if tokens_dict is None:
            tokens_dict = self.tokens_dict
        if tokens_freq is None:
            tokens_freq = self.tokens_freq
        if comments_onehot is None:
            comments_onehot = self.comments_onehot

        ## Adjacency matrix & Features
        check_adj = []
        for i in range(len(tokens_freq)):
            if tokens_freq[i] > 1:
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

        ## Adjacency Matrix

        adj = nx.adj_matrix(G).astype('float32').todense()
        adj = adj + np.eye(G.number_of_nodes(), dtype = 'float32')

        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj = r_mat_inv.dot(adj)

        features = torch.FloatTensor(features)

        # self.G = G
        # self.features = features

        return G, features, adj

    def test_comment_korean(self, test_comment, G = None, nodes_on_feature = None, tokens_dict = None, features = None):
        # if G is None:
        #     G = self.G
        # if nodes_on_feature is None:
        #     nodes_on_feature = self.nodes_on_features
        # if tokens_dict is None:
        #     tokens_dict = self.tokens_dict
        # if features is None:
        #     features = self.features

        test_comment = re.sub(pattern='([ㄱ-ㅎㅏ-ㅣ]+)', repl='', string=test_comment[0])
        test_comment = re.sub(pattern='[^\w\s]', repl=' ', string=test_comment)
        test_comment = test_comment.split(' ')
        if test_comment.count('') > 0:
            for i in range(test_comment.count('')):
                test_comment.remove('')

        test_comment = list(map(tokens_dict.get, test_comment))
        if test_comment.count(None):
            if test_comment.count(None) / len(test_comment) > 0.5:
                print(test_comment.count(None) / len(test_comment),
                      '\n',
                      'The percentage of OOV is over half, thus the prediction may not be precise')
            for i in range(test_comment.count(None)):
                test_comment.remove(None)

        if len(test_comment) == 0:
            print('All tokens are out of vocabulary')

        test_comment_feature = [0] * len(tokens_dict)
        for num in test_comment:
            test_comment_feature[num] += 1

        test_comment_feature = torch.FloatTensor(test_comment_feature).unsqueeze(0)
        features = torch.cat([features, test_comment_feature], 0)

        G.add_node('test')
        for num in test_comment:
            edges = [('test', node) for node in nodes_on_feature[num]]
            G.add_edges_from(edges)

        ## Adjacency Matrix

        adj_test = nx.adj_matrix(G).todense()
        adj_test = adj_test + np.eye(G.number_of_nodes())

        rowsum = np.array(adj_test.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj_test = r_mat_inv.dot(adj_test)

        return features, adj_test

    def test_comment(self, test_comment, language, tokens_dict, features, nodes_on_features):
        if language not in ['en','ko']:
            print('Language should be English or Korean')
            return
        if language == 'en':
            for alphabet in list(string.ascii_lowercase):
                test_comment = re.sub(str('[' + alphabet + ']' + '[' + alphabet + ']' + '[' + alphabet + ']+'), alphabet, test_comment)
                del alphabet

            test_comment = re.sub('[^a-zA-Z]', ' ', test_comment).lower().split()
            stops = set(nltk.corpus.stopwords.words('english'))
            stemmer = nltk.stem.SnowballStemmer('english')

            test_comment = [stemmer.stem(word) for word in test_comment if not word in stops]
            test_comment = list(map(tokens_dict.get, test_comment))

            del stops, stemmer

            if test_comment.count(None):
                if test_comment.count(None) / len(test_comment) > 0.5:
                    print(test_comment.count(None) / len(test_comment),
                          '\n',
                          'The percentage of OOV is over half, thus the prediction may not be precise')
                for i in range(test_comment.count(None)):
                    test_comment.remove(None)

            if len(test_comment) == 0:
                print('All tokens are out of vocabulary')

            test_comment_feature = [0] * features.shape[1]
            for num in test_comment:
                test_comment_feature[num] += 1

            test_comment_feature = torch.FloatTensor(test_comment_feature).unsqueeze(0)
            features = torch.cat([test_comment_feature, features], 0)

            adj_list = torch.zeros([1, features.shape[0]], dtype=torch.float32)
            node_list = [0]
            for num in test_comment:
                node_list.extend(nodes_on_features[num])
            node_list = list(np.unique(node_list))

            for node_num in node_list:
                adj_list[0][node_num] += 1 / len(node_list)

        elif language == 'ko':
            print("준비중")

        return features, adj_list


#coding:utf-8

import numpy as np
import pandas
from config import irConfig
from collections import defaultdict
import json,pickle
import jieba.posseg as pseg


def _load_stop_words():
    f = open("../../dataSet/stop_words.txt", 'rb')
    stop_words = f.readlines()
    f.close()
    stop_words = [ele.strip().decode("utf-8") for ele in stop_words]
    return set(stop_words)

def initialize(path):
    '''
    初始化单词IDF表，倒排索引表。
    :param path: 检索模型数据库路径，txt文件格式是[line1,line2,...]，单词以空格分割
    :return: IDF表：defaultdict,单词-idf值，倒排表：单词-[doc_id:freq,...]
    '''
    stop_words = _load_stop_words()
    vocab = defaultdict(int)
    f = open(path,"rb")
    corpus = f.readlines()
    f.close()
    reverseTable = defaultdict(list)
    for id,line in enumerate(corpus):
        words = line.decode("utf-8").split(" ")
        tmp = defaultdict(int)
        for word in words:
            w,f = word.split("<!>")
            if (not word in stop_words) and (not f in irConfig.stop_flag):
                tmp[w]+=1
        for w,freq in tmp.items():
            vocab[w] += 1
            reverseTable[w].append(id)
        if id%10000 == 0:
            print("processing line %d"%id)

    N = len(corpus)
    idfTable = [(word,np.log((N-freq+0.5)/(freq+0.5))) for word,freq in vocab.items()]
    f = open(irConfig.IDF_PATH,"wb")
    pickle.dump(defaultdict(float,idfTable),f)
    f.close()
    f = open(irConfig.REV_PATH,"wb")
    pickle.dump(reverseTable,f)
    f.close()

def initializeIdfFromPKL(path):
    '''
    从pkl文件中下载idf表,倒排表
    :param path: pkl文件路径
    :return: defaultdict
    '''
    f = open(path,"rb")
    idfTable = pickle.load(f)
    f.close()
    if isinstance(idfTable,dict):
        return defaultdict(float,idfTable)
    else:
        if not isinstance(idfTable,defaultdict):
            raise(TypeError,"idfTable should be instance of dict,but %s detected"%type(idfTable))

def initializeRevFromPKL(path):
    '''
    从pkl文件中下载idf表,倒排表
    :param path: pkl文件路径
    :return: defaultdict
    '''
    f = open(path,"rb")
    revTable = pickle.load(f)
    f.close()
    if isinstance(revTable,dict):
        return defaultdict(list,revTable)
    else:
        if not isinstance(revTable,defaultdict):
            raise(TypeError,"revTable should be instance of dict,but %s detected"%type(revTable))

def index_corpus(path):
    '''
    预处理数据，构建数据库
    :param path: 检索模型数据库路径，txt文件，格式是[line1,line2,...]，单词以空格分割
    :return: pkl文件，存储格式[doc,defaultdict]
    '''
    stop_words = _load_stop_words()
    idfTable = initializeIdfFromPKL(irConfig.IDF_PATH)
    f = open(path,"rb")
    corpus = f.readlines()
    f.close()
    avgl = np.mean([len(line) for line in corpus])
    def _factory(line):
        words = line.decode("utf-8").split(" ")
        alpha0 = irConfig.k1 + 1
        alpha1 = irConfig.k1*(1-irConfig.b+irConfig.b*len(words)/avgl)
        features = defaultdict(int)
        for word in words:
            w,f = word.split("<!>")
            if (not w in stop_words) and (not f in irConfig.stop_flag):
                features[w]+=1
        res = defaultdict(float)
        for word,freq in features.items():
            point_score = idfTable[word]*freq*alpha0/(freq+alpha1)
            res[word] = round(point_score,3)
        return res

    f = open(irConfig.DB_PATH, "w")
    for id,line in enumerate(corpus):
        f.write(json.dumps(_factory(line))+"\n")
        if id%10000 == 0:
            print("processing line %d"%id)
    f.close()



class BM25:
    '''
    BM25算法主类
    '''
    def __init__(self):
        '''
        加载BM25算法用到的数据
        :param Block_Num: 区块编号
        '''
        print("Loading database ...")
        f = open(irConfig.DB_PATH,"rb")
        self.db_features = [json.loads(ele) for ele in f.readlines()]
        f.close()
        f = open("../../dataSet/retrival/db.txt","rb")
        self.data = f.readlines()
        f.close()
        self.idfTable = initializeIdfFromPKL(irConfig.IDF_PATH)
        self.revTable = initializeRevFromPKL(irConfig.REV_PATH)
        self.stop_words = _load_stop_words()
        print("Loaded successfully")
    
    def search(self,Q,k=5):
        '''
        查询相关度最高的DOC
        :param Q: 请求，string
        :return: 字符串
        '''
        words = Q.split(" ")
        features = []
        for word in words:
            w,f = word.split("<!>")
            if (not w in self.stop_words) and (not f in irConfig.stop_flag):
                features.append(w)
        candidates = []
        ids = set()
        scores = []
        for w in features:
            tmp_ids = self.revTable[w]
            for id in tmp_ids:
                if not id in ids:
                    ids.add(id)
                    scores.append(self.calcul_score(self.db_features[id],features))
        # print(len(candidates))
        length = len(ids)
        if length==0:
            return []
        ids = list(ids)
        scores = np.array(scores)
        if length>k:
            inds_sorted = np.argpartition(scores,-k)
        else:
            inds_sorted = np.argsort(scores)
        res = [ids[ind] for ind in inds_sorted[-k:]]
        return res
        # for s,id in res:
        #     ans = self.data[id].decode("utf-8")
        #     print("answer:%s score:%f"%(ans,s))
        #     print(self.db_features[id])
        #     print("=======================")
        # print("next...")

    def calcul_score(self,D, Q):
        '''
        计算文档D和请求Q的得分
        :param D: 带索引的文档，格式为(point_score:defaultdict)
        :param Q: 请求，list
        :return: 得分float
        '''
        point_score = D
        score = 0
        for q in Q:
            try:
                score += point_score[q]
            except:
                pass
        return score

def context_use_BM25():
    model = BM25()
    fw = open("../../dataSet/seq2seq/context.txt","w")
    f = open("../../dataSet/seq2seq/test.txt","rb")
    q,a = f.readline().decode("utf-8").strip(),f.readline().decode("utf-8").strip()
    num = 0
    while q:
        res = model.search(q)
        fw.write(str(res)+"\n")
        num += 1
        if num%10000 == 0:
            print("processing line %d"%num)
        q,a = f.readline().decode("utf-8").strip(),f.readline().decode("utf-8").strip()
    f.close()
    fw.close()


if __name__=="__main__":
    # initialize("../../dataSet/retrival/db.txt")
    # index_corpus("../../dataSet/retrival/db.txt")
    # model = BM25()
    # while True:
    #     query = input("your query:")
    #     model.search(query)
    context_use_BM25()



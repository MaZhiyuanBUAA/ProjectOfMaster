#coding:utf-8

import random
import jieba.posseg as pseg
from collections import defaultdict

def clear_data():
    f=open("data.txt","rb")
    d = f.readlines()
    f.close()
    total = len(d)//2
    print(total)
    f = open("data_cleared.txt","wb")
    num = 0
    for ind in range(total):
        if d[2*ind]==d[2*ind+1]:
            continue
        num += 1
        f.write(d[2*ind])
        f.write(d[2*ind+1])
    print(num)
    f.close()
def pseg_data():
    f = open("data_cleared.txt","rb")
    d = f.readlines()
    f.close()
    f = open("data_pseg.txt","wb")
    for line in d:
        words = pseg.cut(line.decode("utf-8").strip())
        tmp = []
        for w,flag in words:
            tmp.append("<!>".join([w,flag]))
        tmp = " ".join(tmp)+"\n"
        f.write(tmp.encode("utf-8"))
    f.close()

def initialize_vocab():
    f = open("data_pseg.txt","rb")
    d = f.readlines()
    f.close()
    vocab = defaultdict(int)
    for line in d:
        words = line.split(" ")
        for w,f in words.split("<!>"):
            vocab[w] += 1
    vocab = sorted(vocab.items(),key=lambda x:x[1],reverse=True)
    f = open("vocab_freq.txt","wb")
    tmp = []
    for v,freq in vocab:
        tmp.append("<!>".join([v,str(freq)]))
    f.write("\n".join(tmp))
    f.close()

def split_data():
    random.seed(123)
    f = open("data_pseg.txt","rb")
    all_data = f.readlines()
    f.close()
    print(len(all_data))
    total = list(range(len(all_data)//2))
    ts = len(total)

    random.shuffle(total)
    idx0 = total[:int(0.4*ts)]
    idx1 = total[int(0.4*ts):int(0.5*ts)]
    idx2 = total[int(0.5*ts):]

    ftr = open("seq2seq/train.txt","wb")
    fte = open("seq2seq/test.txt","wb")
    for ind in idx0:
        ftr.write(all_data[2*ind])
        ftr.write(all_data[2*ind+1])
    for ind in idx1:
        fte.write(all_data[2*ind])
        fte.write(all_data[2*ind+1])
    ftr.close()
    fte.close()

    fre = open("retrival/db.txt", "wb")
    for ind in idx2:
        fre.write(all_data[2 * ind])
        fre.write(all_data[2 * ind + 1])
    fre.close()

if __name__ == "__main__":
    # clear_data()
    # pseg_data()
    split_data()
#coding:utf-8

class irConfig:
    '''
    检索模型配置文件
    k1:取值范围[1.2,2.0]
    b取0.75
    '''
    k1 = 1.5
    b = 0.75
    BLOCK_SIZE = 500000
    DB_PATH = "data/db_features.txt"
    IDF_PATH = "data/idf.pkl"
    REV_PATH = "data/rev.pkl"
    stop_flag = ['t','x', 'c', 'u', 'p', 'm', 'f', 'r']

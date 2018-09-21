import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import random
import numpy as np
from .custom_konlpy import Okt as Twitter
from gensim.models import Word2Vec as w2v
import params as par
import Util.kor_char_parser as parser

def class_unmapping():
    return list(par.class_map.keys())
classification = par.class_map

class proccessing_util:
    def __init__(self,model=None):
        if model is not None:
            self.model = w2v.load(model)
            self.word2id = {w: i for i, w in enumerate(self.model.wv.index2word, 1)}
            self.twitter = Twitter()
        else:
            pass
            #self.model = w2v.load('model/only_everytime.model')


    def c2v_preprocess(self, data: list, embedding_dim: int, max_length: int = par.max_length):
        """
         입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
         기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
         문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

        :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
        :param max_length: 문자열의 최대 길이
        :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
        """
        vectorized_data = [parser.decompose_str_as_one_hot(datum, warning=False) for datum in data]
        zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
        for idx, seq in enumerate(vectorized_data):
            length = len(seq)
            if length >= max_length:
                length = max_length
                zero_padding[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding[idx, :length] = np.array(seq)

        return zero_padding

    def preprossessing(self,text_list, embedding_dim, maxlen=par.max_length, model=None):
        EMBEDDING_DIM = embedding_dim
        # MAX_SEQ_LENGTH = 50
        if model is not None:
            self.model = model
        embedding_matrix = np.zeros([len(self.word2id) + 1, EMBEDDING_DIM])
        for k in self.word2id:
            embedding_matrix[self.word2id[k]] = self.model.wv[k]


        index_list = []
        for text in text_list:
            tmp_list = []
            # nouns = twitter.nouns(text)
            posses = self.twitter.pos(text, stem=True, norm=True)

            # for noun in nouns:
            #
            #     if noun in word2id.keys():
            #         tmp_list.append(word2id[noun])

            for pos in posses:
                if pos[1] in ['Noun', 'Verb', 'Adjective', 'Adverb', 'Exclamation', 'Determiner'] and pos[
                    0] in self.word2id.keys():
                    tmp_list.append(self.word2id[pos[0]])
            # print(nouns)
            # print(embedding_matrix[tmp_list])
            index_list.append(tmp_list)
        index_list = pad_sequences(index_list, maxlen=maxlen)
        # print(np.shape(embedding_matrix[index_list]))
        return embedding_matrix[index_list]


def c2v_div_dataset(dataframe,embedding_dim,train_size,max_seq_length=par.max_length):

    range_list = np.array(range(dataframe.__len__()))

    train_index = random.sample(range_list.tolist(), k=train_size)
    train_index.sort()
    #print(train_index)

    mask_index = np.ones([dataframe.__len__()])
    mask_index[train_index] = 0

    test_index = range_list[mask_index == 1].tolist()
    text_list = proccessing_util().c2v_preprocess(dataframe['text'], embedding_dim, max_seq_length)
    try:
        class_list = np.array([classification[i] for i in dataframe['class']])
    except:
        return (text_list[train_index], text_list[test_index]) # (train_X, test_X)



    #print(test_index)

    #print(range_list[test_index].tolist())
    return (class_list[train_index],
            text_list[train_index],
            class_list[test_index],
            text_list[test_index],
            [train_index,test_index]
            ) # (train_Y, train_X, test_Y, test_X)

def div_dataset(dataframe,w2v_model,embedding_dim,train_size,max_seq_length=par.max_length):

    range_list = np.array(range(dataframe.__len__()))

    train_index = random.sample(range_list.tolist(), k=train_size)
    train_index.sort()
    #print(train_index)

    mask_index = np.ones([dataframe.__len__()])
    mask_index[train_index] = 0

    test_index = range_list[mask_index == 1].tolist()
    text_list = proccessing_util(w2v_model).preprossessing(dataframe['text'], embedding_dim, max_seq_length)
    try:
        class_list = np.array([classification[i] for i in dataframe['class']])
    except:
        return (text_list[train_index], text_list[test_index]) # (train_X, test_X)



    #print(test_index)

    #print(range_list[test_index].tolist())
    return (class_list[train_index],
            text_list[train_index],
            class_list[test_index],
            text_list[test_index],
            [train_index,test_index]
            ) # (train_Y, train_X, test_Y, test_X)

def div_random_batch(batch_size,dataset_x,dataset_y):
    total_size = len(dataset_x)
    range_list = np.array(range(total_size))
    while total_size < batch_size:
        dataset_x += dataset_x
        dataset_y += dataset_y
        total_size *= 2
    rand_index_list = random.sample(range_list.tolist(),k=batch_size)

    return (dataset_x[rand_index_list],dataset_y[rand_index_list],rand_index_list)

def one_hot(y,num_classes):
    targets = np.array([y]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[targets]
    return one_hot_targets

if __name__ == '__main__':
    print(one_hot([[1,2],[3,4],[5,3],[2,1]],6))
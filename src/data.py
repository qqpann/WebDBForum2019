import os
import pickle
import random
from math import log, ceil
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split


dirname = os.path.join(os.path.dirname(__file__), '../data')
SOURCE_ASSIST0910_SELF = 'selfmade_ASSISTmentsSkillBuilder0910'  # Self made from ASSISTments
SOURCE_ASSIST0910_ORIG = 'original_ASSISTmentsSkillBuilder0910'  # Piech et al.

PAD = 0
SOS = 1


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_source(source) -> List[Tuple[int]]:
    if source == SOURCE_ASSIST0910_SELF:
        filename = os.path.join(dirname, 'input/skill_builder_data_corrected.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    elif source == SOURCE_ASSIST0910_ORIG:
        trainfname = os.path.join(dirname, 'input/builder_train.csv')
        testfname = os.path.join(dirname, 'input/builder_test.csv')
        with open(trainfname, 'r') as f:
            data = []
            lines = f.readlines()
            for idx in range(0, len(lines), 3):
                qlist = list(map(int, lines[idx + 1].strip().rstrip(',').split(',')))
                alist = list(map(int, lines[idx + 2].strip().rstrip(',').split(',')))
                data.append([(q, a) for q, a in zip(qlist, alist)])
    else:
        filename = os.path.join(dirname, f'input/{source}.pickle')
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            data = list(data_dict.values())
    return data


class QandAEmbedder:
    def __init__(self, M: int, sequence_size: int, preserved_tokens:int=2):
        self.M = M
        self.sequence_size = sequence_size
        self.PT = preserved_tokens
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        
    def qaToIdxNum(self, q_and_a: Tuple) -> int:
        '''
        qaToIdxNum(q_ant_a)
        
        Docs
        ----
        (q, a)のタプルを2M+PT長のインデックスに変換する（PTは0:pad, 1:sos）

        Examples
        --------
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.qaToIdxNum((0, 0))
        2
        >>> emb.qaToIdxNum((2, 1))
        7
        >>> emb.qaToIdxNum((3, 0))
        Traceback (most recent call last):
            ...
        ValueError: q out of range
        >>> emb.qaToIdxNum((0, -1))
        Traceback (most recent call last):
            ...
        ValueError: a out of range
        '''
        if not (0 <= q_and_a[0] < self.M):
            raise ValueError('q out of range. Got {0} but expected 0<=q<M={1}'.format(q_and_a, self.M))
        elif (q_and_a[1] not in {0, 1}):
            raise ValueError('a out of range')
            
        # 0: PAD, 1: SOS
        return q_and_a[0] + q_and_a[1] * self.M + self.PT  # consider 0:pad and 1:sos

    def idxToOneHot(self, idx: int) -> np.array:  # idxnum should already considered 0
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.idxToOneHot(0)
        array([1., 0., 0., 0., 0., 0., 0., 0.])
        >>> emb.idxToOneHot(6)
        array([0., 0., 0., 0., 0., 0., 1., 0.])
        >>> emb.idxToOneHot(8)
        Traceback (most recent call last):
            ...
        ValueError: idx out of range
        '''
        if not (0 <= idx < 2 * self.M + self.PT):
            raise ValueError('idx out of range')
        onehot = np.zeros(2 * self.M + self.PT)  # consider 0:pad and 1:sos
        onehot[idx] = 1
        return onehot

    def qaToOnehot(self, q_and_a: Tuple) -> np.array:
        ''' To 2M + PreservedNum Size Onehot '''
        idx = self.qaToIdxNum(q_and_a)
        onehot = self.idxToOneHot(idx)
        return onehot

    def sequenceToOnehot(self, sequence_qa: List) -> List[np.array]:
        length = len(sequence_qa)
        sequence = [self.qaToIdxNum(qa) for qa in sequence_qa] + \
            [self.PAD_IDX] * (self.sequence_size - length)
        onehotSeq = [self.idxToOneHot(idx) for idx in sequence]
        return onehotSeq
    
    def qaToDeltaQ(self, qa:Tuple):
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.qaToDeltaQ((2, 1))
        array([0., 0., 1.])
        '''
        delta_q = np.zeros(self.M)
        delta_q[qa[0]] = 1
        return delta_q
        
    def qaToDeltaQandA(self, qa: Tuple) -> np.array:
        ''' To skill size onehot '''
        delta_q = self.qaToDeltaQ(qa)
        a = qa[1]
        return delta_q, a
    
    def sequenceToProbSeq(self, sequence_qa:List) -> np.array:
        '''
        >>> emb = QandAEmbedder(M=3, sequence_size=4)
        >>> emb.sequenceToProbSeq([(1, 1), (2, 0)])
        array([[0.5, 1. , 0.5],
               [0.5, 1. , 0. ]])
        '''
        base = np.ones(shape=(len(sequence_qa), self.M)) / 2
        for i, qa in enumerate(sequence_qa):
            if i != 0:
                base[i] = base[i-1]
            base[i, qa[0]] = qa[1]
        return base

    
# type: base, encdec, generative
def slice_d(d: List, x_seq_size:int, type:str='base', sliding_window:int=0, reverse:bool=False, extend_backward:int=0, extend_forward:int=0) -> Tuple[List]:
    '''
    Params
    ------
    d:
        list. data sequence.
        
    x_seq_size:
        int.
        
    sliding_window:
        int. if 1, only get 1 result for 1 user.
        if 0, get all for a user.
    
    generative:
        bool. if True, x and y have same length and do not overlap.
        
    reversed:
        bool. if True, get data from latest ones.
    
    Example
    -------
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=1, reverse=False)
    [[0, 1, 2]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=2, reverse=False)
    [[0, 1, 2], [3, 4, 5]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=0, reverse=False)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=9, reverse=False)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=1, reverse=False)
    [[0, 1, 2, 3]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=2, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=0, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=9, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=1, reverse=True)
    [[6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=2, reverse=True)
    [[6, 7, 8], [3, 4, 5]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=0, reverse=True)
    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    >>> slice_d(d, x_seq_size=2, type='base', sliding_window=9, reverse=True)
    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=1, reverse=False)
    [[0, 1, 2, 3]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=0, reverse=False)
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=1, reverse=True)
    [[5, 6, 7, 8]]
    >>> slice_d(d, x_seq_size=2, type='generative', sliding_window=0, reverse=True)
    [[5, 6, 7, 8], [1, 2, 3, 4]]
    
    # >>> d = [0, 1, 2]
    # >>> slice_d(d, x_seq_size=5, type='base', sliding_window=1, reverse=False)
    # [[0, 1, 2, ?, ?]]
    
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> slice_d(d, x_seq_size=2, type='encdec', sliding_window=1, reverse=False, extend_backward=1, extend_forward=1)
    [[0, 1, 2, 3, 4]]
    '''
    result = []
    if type[:4] == 'base':
        seq_size = x_seq_size + 1
    elif type == 'generative':
        seq_size = x_seq_size + x_seq_size
    elif type == 'encdec':
        seq_size = x_seq_size + 2 + extend_forward
    else:
        raise ValueError
    max_iter = len(d) // seq_size
    assert sliding_window >= 0 and sliding_window % 1 == 0
    max_iter = min(max_iter, sliding_window) if sliding_window != 0 else max_iter
    prefix = 0 if not reverse else len(d) - seq_size
    direction = 1 if not reverse else -1
    for i in range(0, direction * max_iter, direction * 1):
        result.append(d[prefix + i * seq_size : prefix + i * seq_size + seq_size])
    return result 
        
    

def split_encdec(xsty_seq:List, extend_backward=0, extend_forward=0) -> (List, List, List):
    '''
    Split x_src, x_trg, y
    
    Examples
    --------
    >>> xsty_seq = [2, 3, 4, 5, 6]
    >>> split_encdec(xsty_seq)
    ([2, 3, 4], [5], [6])
    >>> split_encdec(xsty_seq, 1, 0)
    ([2, 3, 4], [4, 5], [5, 6])
    >>> xsty_seq = [2, 3, 4, 5, 6, 7]
    >>> split_encdec(xsty_seq, 0, 1)
    ([2, 3, 4], [5, 6], [6, 7])
    '''
    x_src = xsty_seq[:-2 - extend_forward]
    x_trg = xsty_seq[-2 - extend_backward - extend_forward:-1]
    y = xsty_seq[-1 - extend_backward - extend_forward:]
    return x_src, x_trg, y
    
    

# ==============================================================================
#       Merge other prepare_?_data into prepare_data
# ==============================================================================
def prepare_data(source, type, n_skills, preserved_tokens, min_n, max_n, batch_size, device, sliding_window:int=1, params={}): #TODO: fix sw
    assert type in {'base', 'encdec'}
    data = load_source(source)
        
    M = n_skills
    sequence_size = max_n
    N = ceil(log(2 * M))
    
    qa_emb = QandAEmbedder(M, sequence_size)

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(data, [train_num, len(data) - train_num])
    
    if type == 'encdec':
        def get_ds(data):
            x_src_indexed = []
            x_trg_indexed = []
            y_indexed = []
            y_delta_q = []
            y_a = []
            y_prob_qa = []
            for d in data:
                if len(d) < sequence_size + 1 + params.get('extend_forward', 0):
                    continue
                for xsty_seq in slice_d(d, x_seq_size=sequence_size, type=type, sliding_window=sliding_window, reverse=True, **params):
                    # Because it is not generative...
                    # y is the last one
                    x_src, x_trg, y = split_encdec(xsty_seq, **params)
                    x_src_indexed.append([qa_emb.qaToIdxNum(qa) for qa in x_src])
                    x_trg_indexed.append([qa_emb.qaToIdxNum(qa) for qa in x_trg])
                    y_indexed.append([qa_emb.qaToIdxNum(qa) for qa in y]) # コメント時効？→embedding済みは学習対象にしても仕方がない（か？）
                    y_delta_q.append([qa_emb.qaToDeltaQ(qa) for qa in y])
                    y_a.append([qa[1] for qa in y])
                    # yで必要なのは確率分布124と、delta q, aのパターン
                    # delta qから確率分布124を作成する
                    y_prob_qa.append(qa_emb.sequenceToProbSeq(y))


            all_ds = TensorDataset(
                torch.LongTensor(x_src_indexed).to(device), 
                torch.LongTensor(x_trg_indexed).to(device), 
                torch.LongTensor(y_indexed).to(device), 
                torch.Tensor(y_delta_q).to(device), 
                torch.Tensor(y_a).to(device), 
                torch.tensor(y_prob_qa).to(device),
            )
            return all_ds
    else:
        def get_ds(data):
            x_values = []
            y_values = []
            x_onehot = []
            y_onehot = []
            y_onehot_q = []
            y_onehot_a = []
            for d in data:
                if len(d) < sequence_size + 1 + params.get('extend_forward', 0):
                    continue
                for xy_seq in slice_d(d, x_seq_size=sequence_size, type=type, sliding_window=sliding_window, reverse=True):
                    # Because it is not generative...
                    # y is the last one
                    x_values.append(xy_seq[:-1])
                    y_values.append(xy_seq[-1])
                    x_onehot.append(qa_emb.sequenceToOnehot(xy_seq[:-1]))
                    y_onehot.append(qa_emb.qaToOnehot(xy_seq[-1]))
                    delta_q, a = qa_emb.qaToDeltaQandA(xy_seq[-1])
                    y_onehot_q.append(delta_q)
                    y_onehot_a.append(a)

            x_tensor, y_tensor = torch.Tensor(x_onehot).to(device), torch.Tensor(y_onehot).to(device)
            y_tensor_q, y_tensor_a = torch.Tensor(y_onehot_q).to(device), torch.Tensor(y_onehot_a).to(device)
            all_ds = TensorDataset(x_tensor, y_tensor_q, y_tensor_a)
            return all_ds
    
    
    train_ds = get_ds(train_data)
    eval_ds = get_ds(eval_data)

    # all_dl = DataLoader(all_ds, batch_size=batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, drop_last=True)
    return train_dl, eval_dl



# type: base, encdec, generative
def slide_d(d: List, x_seq_size:int, type:str='base') -> Tuple[List]:
    '''
    Params
    ------
    d:
        list. data sequence.
        
    x_seq_size:
        int.
    
    Example
    -------
    >>> d = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> slide_d(d, x_seq_size=2, type='encdec')
    [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    '''
    result = []
    if type == 'base':
        seq_size = x_seq_size + 1
    elif type == 'generative':
        seq_size = x_seq_size + x_seq_size
    elif type == 'encdec':
        seq_size = x_seq_size + 2
    else:
        raise ValueError
    max_iter = len(d) - seq_size + 1
    prefix = 0
    direction = 1
    for i in range(0, direction * max_iter, direction * 1):
        result.append(d[prefix + i : prefix + i + seq_size])
    return result 


# ==============================================================================
#       Merge prepare_heatmap_rnn_data into prepare_heatmap_data
# ==============================================================================
def prepare_heatmap_data(source, type, n_skills, preserved_tokens, min_n, max_n, batch_size, device, sliding_window:int=1, params={}): #TODO: fix sw
    assert type in {'base', 'encdec', 'baselstm', 'basernn'}
    data = load_source(source)

    M = n_skills
    sequence_size = max_n
    N = ceil(log(2 * M))
    
    qa_emb = QandAEmbedder(M, sequence_size)

    if type == 'encdec':
        def get_ds(data):
            x_src_indexed = []
            x_trg_indexed = []
            y_indexed = []
            y_delta_q = []
            y_a = []
            y_prob_qa = []
            for d in data:
                if len(d) <= sequence_size or len(d) < 80:
                    continue
                for xsty_seq in slice_d(d, x_seq_size=sequence_size, type=type, sliding_window=sliding_window, reverse=True, **params):
                    # Because it is not generative...
                    # y is the last one
                    x_src, x_trg, y = split_encdec(xsty_seq, **params)
                    x_src_indexed.append([qa_emb.qaToIdxNum(qa) for qa in x_src])
                    x_trg_indexed.append([qa_emb.qaToIdxNum(qa) for qa in x_trg])
                    y_indexed.append([qa_emb.qaToIdxNum(qa) for qa in y]) # コメント時効？→embedding済みは学習対象にしても仕方がない（か？）
                    y_delta_q.append([qa_emb.qaToDeltaQ(qa) for qa in y])
                    y_a.append([qa[1] for qa in y])
                    # yで必要なのは確率分布124と、delta q, aのパターン
                    # delta qから確率分布124を作成する
                    y_prob_qa.append(qa_emb.sequenceToProbSeq(y))


            all_ds = TensorDataset(
                torch.LongTensor(x_src_indexed).to(device), 
                torch.LongTensor(x_trg_indexed).to(device), 
                torch.LongTensor(y_indexed).to(device), 
                torch.Tensor(y_delta_q).to(device), 
                torch.Tensor(y_a).to(device), 
                torch.tensor(y_prob_qa).to(device),
            )
            return all_ds
    else:
        def get_ds(data):
            x_values = []
            y_values = []
            x_onehot = []
            y_onehot = []
            y_onehot_q = []
            y_onehot_a = []
            for d in data:
                if len(d) <= sequence_size or len(d) < 80:
                    continue
                for xy_seq in slice_d(d, x_seq_size=sequence_size, type=type, sliding_window=sliding_window, reverse=True):
                    # Because it is not generative...
                    # y is the last one
                    x_values.append(xy_seq[:-1])
                    y_values.append(xy_seq[-1])
                    x_onehot.append(qa_emb.sequenceToOnehot(xy_seq[:-1]))
                    y_onehot.append(qa_emb.qaToOnehot(xy_seq[-1]))
                    delta_q, a = qa_emb.qaToDeltaQandA(xy_seq[-1])
                    y_onehot_q.append(delta_q)
                    y_onehot_a.append(a)

            x_tensor, y_tensor = torch.Tensor(x_onehot).to(device), torch.Tensor(y_onehot).to(device)
            y_tensor_q, y_tensor_a = torch.Tensor(y_onehot_q).to(device), torch.Tensor(y_onehot_a).to(device)
            all_ds = TensorDataset(x_tensor, y_tensor_q, y_tensor_a)
            return all_ds
    
    
    ds = get_ds(data)

    # all_dl = DataLoader(all_ds, batch_size=batch_size, drop_last=True)
    dl = DataLoader(ds, batch_size=batch_size, drop_last=True)
    return dl




# ========================= Seq2Seq not used anymore ======================================

def split_xy_seq(xy_seq:List, generative:bool=False):
    '''
    >>> xy_seq = [2, 3, 4, 5]
    >>> split_xy_seq(xy_seq, generative=False)
    ([2, 3, 4], [3, 4, 5])
    '''
    x_seq = xy_seq[:-1]
    y_seq = xy_seq[1:]
    return x_seq, y_seq


def prepare_seq2seq_data(source, n_skills, preserved_tokens, min_n, max_n, batch_size, device, sliding_window:int=1): #TODO: fix sw
    data = load_source(source)

    M = n_skills
    sequence_size = max_n
    N = ceil(log(2 * M))
    
    qa_emb = QandAEmbedder(M, sequence_size)

    train_num = int(len(data) * .8)
    train_data, eval_data = random_split(data, [train_num, len(data) - train_num])
    
    
    def get_ds(data):
        x_qanda = []
        y_qanda = []
        x_indexed = []
        y_indexed = []
        y_delta_q = []
        y_a = []
        y_prob_qa = []
        for d in data:
            for xy_seq in slice_d(d, x_seq_size=sequence_size, type='base', sliding_window=sliding_window, reverse=True):
                # Because it is not generative...
                # y is the last one
                x_seq, y_seq = split_xy_seq(xy_seq)
                x_qanda.append(x_seq)
                y_qanda.append(y_seq)
                # x_indexed.append([SOS] + [qa_emb.qaToIdxNum(qa) for qa in x_seq])
                # y_indexed.append([SOS] + [qa_emb.qaToIdxNum(qa) for qa in y_seq]) # embedding済みは学習対象にしても仕方がない（か？）
                x_indexed.append([qa_emb.qaToIdxNum(qa) for qa in x_seq])
                y_indexed.append([qa_emb.qaToIdxNum(qa) for qa in y_seq]) # embedding済みは学習対象にしても仕方がない（か？）
                y_delta_q.append([qa_emb.qaToDeltaQ(qa) for qa in y_seq])
                y_a.append([qa[1] for qa in y_seq])
                # yで必要なのは確率分布124と、delta q, aのパターン
                # delta qから確率分布124を作成する
                y_prob_qa.append(qa_emb.sequenceToProbSeq(y_seq))

        # x_tensor, y_tensor = torch.Tensor(x_onehot).to(device), torch.Tensor(y_onehot).to(device)
        # print(x_tensor.shape)
        # torch.Size([1726, 15, 249])
        # torch.Size([440, 15, 249])
        x_tensor, y_tensor = torch.LongTensor(x_indexed).to(device), torch.LongTensor(y_indexed).to(device)
        y_delta_q_tensor, y_a_tensor = torch.Tensor(y_delta_q).to(device), torch.Tensor(y_a).to(device)
        y_prob_tensor = torch.tensor(y_prob_qa).to(device)
        print(x_tensor.shape)
        all_ds = TensorDataset(x_tensor, y_tensor, y_delta_q_tensor, y_a_tensor, y_prob_tensor)
        return all_ds
    
    
    train_ds = get_ds(train_data)
    eval_ds = get_ds(eval_data)

    # all_dl = DataLoader(all_ds, batch_size=batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, drop_last=True)
    return train_dl, eval_dl



if __name__ == '__main__':
    import doctest
    doctest.testmod()
import numpy as np
import re
import pickle


def load_table():
    ch2id = pickle.load(open('data/ch2id.pkl', 'rb'))
    spell2id = pickle.load(open('data/spell2id.pkl', 'rb'))

    return ch2id, spell2id


def load_arr(file_clean, len_thresh):
    '''

    :param file_clean: 准备好的文本文件，三列分别是id, 拼音, 对其的汉字
    :param len_thresh: (min_len, max_len)
    :return: X, Y, dtype: List(ndarray)
    '''
    ch2id, spell2id = load_table()
    X, Y = list(), list()

    with open(file_clean, 'r', encoding='utf-8') as fd:
        for line in fd.readlines():
            idx, spells_s, ch_s = line.split('\t')
            # 把标点替换成'|'，并把长句子分割成短句子的列表，原长句仍然保留
            spells_l = re.sub("(?<=([。，！？]))", '|', spells_s).split('|')
            ch_l = re.sub("(?<=([。，！？]))", '|', ch_s).split('|')

            for spells_s, ch_s in zip(spells_l + [spells_s], ch_l + [ch_s]):
                spells_s, ch_s = spells_s.strip(), ch_s.strip()
                assert len(spells_s) == len(ch_s)
                if len_thresh[0] <= len(spells_s) <= len_thresh[1]:
                    # 映射表中[1]对应着Unknown
                    x = [spell2id.get(spell, 1) for spell in spells_s]
                    y = [ch2id.get(ch, 1) for ch in ch_s]

                    n_pad = len_thresh[1] - len(x)
                    x += [0 for _ in range(n_pad)]  # 0对应着Empty
                    y += [0 for _ in range(n_pad)]

                    X.append(x)
                    Y.append(y)

    return np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)


class Dataset:
    def __init__(self, data_path, batch_size=32, len_thresh=(10, 50), shuffle=False):
        self.data = list()
        self.target = list()
        self._n_samples = 0
        self.n_features = 0
        self._len_thresh = len_thresh  # 句子长度范围

        self._idx = 0  # mini-batch的游标
        self._batch_size = batch_size

        self._load(data_path)

        if shuffle:
            self._shuffle_data()

        print(self.data.shape, self.target.shape)

    def _load(self, data_path):
        '''
        载入数据
        '''
        self.data, self.target = load_arr(data_path, self._len_thresh)
        self._n_samples, self.n_features = self.data.shape[0], self.data.shape[1]

    def _shuffle_data(self):
        '''
        打乱数据
        '''
        idxs = np.random.permutation(self._n_samples)
        self.data = self.data[idxs]
        self.target = self.target[idxs]

    def next_batch(self):
        '''
        生成mini-batch
        '''
        while self._idx + self._batch_size < self._n_samples:
            yield self.data[self._idx: (self._idx + self._batch_size)], \
                  self.target[self._idx: (self._idx + self._batch_size)]
            self._idx += self._batch_size

        self._idx = 0
        self._shuffle_data()


if __name__ == '__main__':
    train_data = Dataset('data/data_clean.txt')
    eval_data = Dataset('eval/eval_clean.txt')
    del train_data, eval_data

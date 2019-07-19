from xpinyin import Pinyin
import regex as re


def clean(text):
    '''
    句子级别的清洗，删除含字母或数字的句子，并将所有非法字符替换成' '
    :param text: 中文文本
    :return: 清洗之后的文本
    '''
    # 跳过包含字母与数字的句子
    if re.search("[A-Za-z0-9]", text) is not None:
        return ''

    text = re.sub("[^ \p{Han}。，！？]", " ", text)  # 非汉字与非[。，！？]字符全替换成空格
    return text


def zip_spell_chs(chs, spell_transformer):
    '''
    将汉语句子转成(拼音, 汉字)的元组，并使用'_'对齐
    :param chs: 单个汉语句子
    :param spell_transformer: 拼音转换器
    :return: (拼音, 汉字)的元组
    '''
    spell_l = spell_transformer.get_pinyin(chs, ' ').split()
    chs_l = list()
    for ch, spell in zip(chs.replace(' ', ''), spell_l):  # 去除汉语句子中的所有空格
        chs_l.extend([ch] + ['_'] * (len(spell) - 1))  # 拼音比汉字多的位用'_'填充

    spells, chs = ''.join(spell_l), ''.join(chs_l)
    return spells, chs


def preprocess(file_I, file_O, spell_transformer):
    '''
    对源文件预处理
    :param file_I: 输入文本文件
    :param file_O: 输出文本文件
    :param spell_transformer: 拼音转换器
    :return:
    '''
    with open(file_O, 'w', encoding='utf-8') as fd_O:  # Outputs
        with open(file_I, 'r', encoding='utf-8') as fd_I:  # Inputs
            for line in fd_I.readlines():
                idx, text = line.split('\t')
                text = clean(text)

                if len(text) > 0:
                    spells, chs = zip_spell_chs(text, spell_transformer)
                    fd_O.write('{}\t{}\t{}\n'.format(idx, spells, chs))

    del spell_transformer


from collections import Counter
from itertools import chain
import pickle


def build_vocab():
    '''
    构建映射表
    :return:
    '''
    spell_vals = 'EUabcdefghijklmnopqrstuvwxyz0123456789。，！？'  # E: Empty, U: Unknown
    spell2id = {spell: idx for idx, spell in enumerate(spell_vals)}
    id2spell = {idx: spell for idx, spell in enumerate(spell_vals)}

    cnt_thresh = 5  # 汉字出现频率阈值
    tmp = [line.split('\t')[2].strip()
           for line in open('data/data_clean.txt', 'r', encoding='utf-8').readlines()]
    chs_cnt = Counter(chain.from_iterable(tmp))
    ch_val = [ch for ch, cnt in chs_cnt.items() if cnt > cnt_thresh]

    ch_val.remove('_')  # 移除统计的'_'
    ch_val = ['E', 'U', '_'] + ch_val  # 将三个特殊字符放在表头
    ch2id = {ch: idx for idx, ch in enumerate(ch_val)}
    id2ch = {idx: ch for idx, ch in enumerate(ch_val)}

    assert ch2id['E'] == 0 and ch2id['U'] == 1 and ch2id['_'] == 2 and \
           id2ch[0] == 'E' and id2ch[1] == 'U' and id2ch[2] == '_'

    assert spell2id['E'] == 0 and spell2id['U'] == 1 and \
           id2spell[0] == 'E' and id2spell[1] == 'U'

    assert len(ch2id) == len(id2ch) and len(spell2id) == len(id2spell)

    pickle.dump(spell2id, open('data/spell2id.pkl', 'wb'))
    pickle.dump(id2spell, open('data/id2spell.pkl', 'wb'))
    pickle.dump(ch2id, open('data/ch2id.pkl', 'wb'))
    pickle.dump(id2ch, open('data/id2ch.pkl', 'wb'))


if __name__ == '__main__':
    # 训练集的处理
    print('preprocessing...')
    file_I = 'data/zho_news_2007-2009_1M-sentences.txt'
    file_O = 'data/data_clean.txt'
    preprocess(file_I, file_O, Pinyin())  # 对源文件做预处理

    print('buliding table...')
    build_vocab()  # 构建映射表

    # 测试集的处理
    print('preprocessing...')
    file_I = 'eval/eval.txt'
    file_O = 'eval/eval_clean.txt'
    preprocess(file_I, file_O, Pinyin())  # 对源文件做预处理

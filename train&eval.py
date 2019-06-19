from dataset import load_table, Dataset

import tensorflow as tf
from model import Network

import distance
import pickle
import numpy as np

if __name__ == '__main__':
    # 该配置下大概需要4G显存
    ch2id, spell2id = load_table()
    voc_size = len(ch2id)

    batch_size = 32
    len_thresh = (10, 50)  # 长度阈值
    t_size = len_thresh[1]

    train_data = Dataset('data/data_clean.txt', batch_size,
                         len_thresh, shuffle=True)
    test_data = Dataset('eval/eval_clean.txt', batch_size,
                        len_thresh, shuffle=True)

    tf.reset_default_graph()
    model = Network(voc_size)

    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(model.init)
        epochs = 1  # 太慢了，只跑一次
        batch_cnt = 0
        for epoch in range(epochs):
            for batch_data, batch_labels in train_data.next_batch():
                batch_cnt += 1
                loss_val, acc_val, _ = sess.run([model.loss, model.acc, model.train_op],
                                                feed_dict={model.X: batch_data, model.Y: batch_labels,
                                                           model.is_training: True})

                if batch_cnt % 500 == 0:
                    print('epoch: {}, batch_loss: {}, batch_acc: {}'
                          .format(epoch + 1, loss_val, acc_val))

                if batch_cnt % 3000 == 0:
                    test_acc_val = sess.run(model.acc, feed_dict={model.X: test_data.data,
                                                                  model.Y: test_data.target,
                                                                  model.is_training: False})
                    print('epoch: {}, test_acc: {}'.format(epoch + 1, test_acc_val))

        Y_pred = sess.run(model.preds, feed_dict={model.X: test_data.data, model.Y: test_data.target,
                                                  model.is_training: False})

    id2ch = pickle.load(open('data/id2ch.pkl', 'rb'))

    with open('eval/eval_res.csv', 'w', encoding='utf-8') as fd:
        fd.write('True,Pred,CER\n')
        total_cer = 0

        for y_test, y_pred in zip(test_data.target, Y_pred):
            s_len = np.count_nonzero(y_test)
            y_test_ch = ''.join([id2ch[idx]
                                 for idx in y_test])[:s_len].replace('_', '')
            y_pred_ch = ''.join([id2ch[idx]
                                 for idx in y_pred])[:s_len].replace('_', '')
            cer = distance.levenshtein(y_test_ch, y_pred_ch) / s_len

            fd.write('{},{},{:.2f}\n'.format(y_test_ch, y_pred_ch, cer))

            total_cer += cer

        fd.write('Total CER: {:.2f}\n'.format(total_cer / test_data.target.shape[0]))

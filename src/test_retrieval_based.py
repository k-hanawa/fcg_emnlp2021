import argparse

import numpy as np
import six

import chainer

import net
import utils

import os
import json
from more_itertools import chunked
from tqdm import tqdm

import pandas as pd


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL')
    parser.add_argument('SETTING')
    parser.add_argument('--topk', '-k', type=int, default=5)
    parser.add_argument('--train-data', '-t')
    parser.add_argument('--test-data', '-v')
    parser.add_argument('--out', '-o')
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()

    with open(args.SETTING) as fi:
        setting = json.load(fi)

    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.src.json')) as fi:
        vocab_source = json.load(fi)
    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.tgt.json')) as fi:
        vocab_target = json.load(fi)

    with open(args.train_data) as fi:
        train_data_raw = [l.rstrip('\n').split('\t') for l in fi]
    with open(args.test_data) as fi:
        test_data_raw = [l.rstrip('\n').split('\t') for l in fi]
    train_source, train_target, train_s_t, train_oovs, train_offset = utils.load_dataset(args.train_data, vocab_source,
                                                                                          vocab_target)
    test_source, test_target, test_s_t, test_oovs, test_offset = utils.load_dataset(args.test_data, vocab_source, vocab_target)

    assert len(train_source) == len(train_target)
    train_data = [(s, t, st, v, o) for s, t, st, v, o in six.moves.zip(
        train_source, train_target, train_s_t, train_oovs, train_offset)]

    assert len(test_source) == len(test_target)
    test_data = [(s, t, st, v, o) for s, t, st, v, o in six.moves.zip(
        test_source, test_target, test_s_t, test_oovs, test_offset)]

    model = getattr(net, setting['modelname'])(
        n_vocab_source=len(vocab_source),
        n_vocab_target=len(vocab_target),
        n_encoder_layers=setting.get('encoder_layer', None),
        n_emb_source=setting.get('n_emb_source', None),
        n_emb_target=setting.get('n_emb_target', None),
        n_encoder_units=setting.get('encoder_unit', None),
        n_encoder_dropout=0,
        n_decoder_units=setting.get('decoder_unit', None)
    )

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    chainer.serializers.load_npz(args.MODEL, model)

    train_agedas = []
    for batch in chunked(tqdm(train_data), args.batchsize):
        inputs, target, s_t, oovs = utils.convert(
            batch,
            args.gpu
        )
        agenda, hxs = model.encode(*inputs)
        train_agedas.append(chainer.cuda.to_cpu(agenda.data))
    train_ageda_vec = np.vstack(train_agedas)

    test_agedas = []
    for batch in chunked(tqdm(test_data), args.batchsize):
        inputs, target, s_t, oovs = utils.convert(
            batch,
            args.gpu
        )
        agenda, hxs = model.encode(*inputs)
        test_agedas.append(chainer.cuda.to_cpu(agenda.data))
    test_ageda_vec = np.vstack(test_agedas)

    nn_idxs = []
    sims = []

    normalized_train_agenda = train_ageda_vec / np.reshape(np.linalg.norm(train_ageda_vec, axis=1),
                                                           (train_ageda_vec.shape[0], 1))
    for agenda, test in tqdm(zip(test_ageda_vec, test_data_raw)):
        cos_dists = np.dot(normalized_train_agenda, agenda) / np.linalg.norm(agenda)
        nn_idx = np.argsort(cos_dists)[::-1][:args.topk]
        nn_idxs.append(nn_idx)
        sims.append(cos_dists[nn_idx])

    refs = []
    hyps = []
    outs = []
    for nn_idx, sim, raw_d in tqdm(zip(nn_idxs, sims, test_data_raw)):
        tgt = raw_d[3]
        ret = train_data_raw[nn_idx[0]]
        ret_tgt = ret[3]

        refs.append(tgt)
        hyps.append(ret_tgt)

        o = []
        colums = []
        o.append('{}'.format(raw_d[0]))
        colums.append('ID')

        o.append('{}'.format(raw_d[1]))
        colums.append('input')

        o.append('{}'.format(raw_d[2]))
        colums.append('offset')

        for i in range(len(nn_idx)):

            ret = train_data_raw[nn_idx[i]]
            o.append('{}'.format(nn_idx[i]))
            colums.append('ret_i {}'.format(i))

            o.append('{}'.format(sim[i]))
            colums.append('emb cos sim {}'.format(i))

            o.append('{}'.format(ret[0]))
            colums.append('ret_ID {}'.format(i))
            o.append('{}'.format(ret[1]))
            colums.append('ret input {}'.format(i))
            o.append('{}'.format(ret[2]))
            colums.append('ret offset {}'.format(i))
            o.append('{}'.format(ret[3]))
            colums.append('output {}'.format(i))

        o.append('{}'.format(raw_d[3]))
        colums.append('gold')

        outs.append(o)

    df = pd.DataFrame(outs, columns=colums)
    df.to_csv(args.out, encoding='utf-16', sep='\t')

if __name__ == '__main__':
    main()

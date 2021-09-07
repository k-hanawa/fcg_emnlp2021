import argparse

import six

import chainer

import net
import utils

import os
import json
from tqdm import tqdm
import pandas as pd

def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser(description='Attention-based NMT')
    parser.add_argument('MODEL', help='vocabulary file')
    parser.add_argument('SETTING', help='vocabulary file')
    parser.add_argument('--test-data', '-v', help='source sentence list')
    parser.add_argument('--metric', '-m', default='cos', help='source sentence list')
    parser.add_argument('--pred-err-model', '-p', help='source sentence list')
    parser.add_argument('--out', '-o', help='source sentence list')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()


    with open(args.SETTING) as fi:
        setting = json.load(fi)

    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.src.json')) as fi:
        vocab_source = json.load(fi)
    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.tgt.json')) as fi:
        vocab_target = json.load(fi)

    vocab_target_i = {v: k for k, v in vocab_target.items()}

    with open(args.test_data) as fi:
        test_data_raw = [l.rstrip('\n').split('\t') for l in fi]
    test_source, test_target, test_s_t, test_oovs, test_offset = utils.load_dataset(args.test_data, vocab_source, vocab_target)

    assert len(test_source) == len(test_target)
    test_data = [(s, t, st, v, o) for s, t, st, v, o in six.moves.zip(
        test_source, test_target, test_s_t, test_oovs, test_offset)]

    model = getattr(net, setting['modelname'])(
        n_vocab_source=len(vocab_source),
        n_vocab_target=len(vocab_target),
        n_encoder_layers=setting['encoder_layer'],
        n_emb_source=setting['n_emb_source'],
        n_emb_target=setting['n_emb_target'],
        n_emb_label=setting.get('n_emb_label', None),
        n_encoder_units=setting['encoder_unit'],
        n_encoder_dropout=setting['encoder_dropout'],
        n_decoder_units=setting['decoder_unit']
    )

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    chainer.serializers.load_npz(args.MODEL, model)

    outs = []
    for test, raw_d in zip(tqdm(test_data), test_data_raw):
        inputs, target, s_t, oovs = utils.convert(
            [test],
            args.gpu
        )
        result = model.predict_multiple(inputs, s_t, 100, 5)[0]

        max_res = (-9999,)
        for r in result:
            if r[0] > max_res[0]:
                max_res = r

        def id2word_oov(i, oov, vocab):
            if i >= len(vocab):
                oi = i - len(vocab)
                if oi < len(oov):
                    w = oov[oi]
                else:
                    w = '<UNK:%d,%d>' % (oi, i)
                return w
            else:
                return vocab[i]

        result_sentence = ' '.join([id2word_oov(
            int(y), oovs[0], vocab_target_i) for y in max_res[1] if int(y) != utils.EOS])

        si_s, ei_s, *err_label = raw_d[2].split(' ')
        o = []
        colums = []
        o.append('{}'.format(raw_d[0]))
        colums.append('ID')

        o.append('{}'.format(raw_d[1]))
        colums.append('input')

        o.append('{} {}'.format(si_s, ei_s))
        colums.append('offset')

        if len(err_label) == 1:
            o.append('{}'.format(err_label[0]))
            colums.append('error label')

        o.append('{:.4f}'.format(max_res[0]))
        colums.append('log prob 0')

        normalized_lprob = max_res[0] / (len(result_sentence.split(' ')) + 1)
        o.append('{:.4f}'.format(normalized_lprob))
        colums.append('normalized log prob 0')

        o.append('{}'.format(result_sentence))
        colums.append('output 0')
        o.append('{}'.format(raw_d[3]))
        colums.append('gold')

        outs.append(o)


    df = pd.DataFrame(outs, columns=colums)
    df.to_csv(args.out, encoding='utf-16', sep='\t')


if __name__ == '__main__':
    main()

"""Train a seq2seq model."""
import argparse

import six

import chainer

import net
import utils

import os
import json

from tqdm import tqdm
import pandas as pd

from collections import OrderedDict



def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL')
    parser.add_argument('SETTING')
    parser.add_argument('--test-data', '-v')
    parser.add_argument('--ret-n', '-n', default='0')
    parser.add_argument('--out', '-o')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()


    with open(args.SETTING) as fi:
        setting = json.load(fi)

    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.src.json')) as fi:
        vocab_source = json.load(fi)
    with open(os.path.join(os.path.dirname(args.SETTING), 'vocab.tgt.json')) as fi:
        vocab_target = json.load(fi)

    vocab_source_i = {v: k for k, v in vocab_source.items()}
    vocab_target_i = {v: k for k, v in vocab_target.items()}

    copy_source = [int(s) for s in setting.get('copy_source', '0,1,2').split(',')]

    model = getattr(net, setting['modelname'])(
        n_vocab_source=len(vocab_source),
        n_vocab_target=len(vocab_target),
        n_encoder_layers=setting['encoder_layer'],
        n_emb_source=setting['n_emb_source'],
        n_emb_target=setting['n_emb_target'],
        n_encoder_units=setting['encoder_unit'],
        n_encoder_dropout=setting['encoder_dropout'],
        n_decoder_units=setting['decoder_unit'],
        n_attention_units=0,
        copy_source=copy_source
    )

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    chainer.serializers.load_npz(args.MODEL, model)

    test_data_raw = pd.read_csv(args.test_data, encoding='utf-16', sep='\t', index_col=0)

    refs = []
    hyps = []
    all_results = OrderedDict()
    raw_ds = OrderedDict()

    for i_ret in args.ret_n.split(','):
        i_ret = int(i_ret)
        test_dataset \
            = utils.load_dataset_with_ret(args.test_data,
                                          vocab_source,
                                          vocab_target,
                                          ret_rank=i_ret)
        test_data = list(
            six.moves.zip(*test_dataset)
        )

        for test, _raw_data in zip(tqdm(test_data), test_data_raw.iterrows()):
            raw_data = [_raw_data[1]]
            inputs, target, s_t, oovs, *_ = utils.convert_with_ret(
                [test],
                args.gpu
            )
            results = model.predict_multiple(inputs, s_t, 100, 5)

            for idx, (result, raw_d) in enumerate(zip(results, raw_data)):
                max_res = (-9999, )
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
                hyps.append(result_sentence)
                refs.append(raw_d['gold'])
                if raw_d['ID'] not in all_results:
                    all_results[raw_d['ID']] = []
                all_results[raw_d['ID']].append((max_res, result_sentence))
                raw_ds[raw_d['ID']] = raw_d

    outs = []
    for _id in all_results:
        raw_d = raw_ds[_id]
        o = []
        colums = []
        o.append('{}'.format(raw_d['ID']))
        colums.append('ID')

        o.append('{}'.format(raw_d['input']))
        colums.append('input')

        o.append('{}'.format(raw_d['offset']))
        colums.append('offset')

        o.append('{}'.format(raw_d['gold']))
        colums.append('gold')

        for i_ret, (max_res, result_sentence) in zip(args.ret_n.split(','), all_results[_id]):
            i_ret = int(i_ret)

            o.append('{}'.format(raw_d['ret_i {}'.format(i_ret)]))
            colums.append('ret_i {}'.format(i_ret))
            o.append('{}'.format(raw_d['ret_ID {}'.format(i_ret)]))
            colums.append('ret_ID {}'.format(i_ret))

            o.append('{}'.format(raw_d['emb cos sim {}'.format(i_ret)]))
            colums.append('emb cos sim {}'.format(i_ret))

            o.append('{}'.format(raw_d['ret input {}'.format(i_ret)]))
            colums.append('ret input {}'.format(i_ret))
            o.append('{}'.format(raw_d['ret offset {}'.format(i_ret)]))
            colums.append('ret offset {}'.format(i_ret))
            o.append('{}'.format(raw_d['output {}'.format(i_ret)]))
            colums.append('ret output {}'.format(i_ret))

            o.append('{:.4f}'.format(max_res[0]))
            colums.append('log prob {}'.format(i_ret))

            normalized_lprob = max_res[0] / (len(result_sentence.split(' ')) + 1)
            o.append('{:.4f}'.format(normalized_lprob))
            colums.append('normalized log prob {}'.format(i_ret))

            o.append('{}'.format(result_sentence))
            colums.append('output {}'.format(i_ret))

        outs.append(o)

    df = pd.DataFrame(outs, columns=colums)
    df.to_csv(args.out, encoding='utf-16', sep='\t')


if __name__ == '__main__':
    main()

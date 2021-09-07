import argparse

import six

import chainer
from chainer import training
from chainer.training import extensions

import net
import utils

import os
import json
import datetime

import random
import numpy as np

def main():
    current_datetime = '{}'.format(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument('DATA')
    parser.add_argument('VOCAB_SRC')
    parser.add_argument('VOCAB_TGT')
    parser.add_argument('--modelname', '-m', default='Seq2Seq')
    parser.add_argument('--val')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--n-emb-source', type=int, default=200)
    parser.add_argument('--n-emb-target', type=int, default=200)
    parser.add_argument('--n-emb-label', type=int, default=200)
    parser.add_argument('--encoder-unit', type=int, default=300)
    parser.add_argument('--encoder-layer', type=int, default=1)
    parser.add_argument('--encoder-dropout', type=float, default=0.1)
    parser.add_argument('--decoder-unit', type=int, default=300)
    parser.add_argument('--min-source-sentence', type=int, default=1)
    parser.add_argument('--max-source-sentence', type=int, default=100)
    parser.add_argument('--out', '-o', default='result')
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    random.seed(args.seed)
    np.random.seed(args.seed)
    chainer.cuda.cupy.random.seed(args.seed)

    vocab_source, emb_source = utils.get_vocab(args.VOCAB_SRC)
    vocab_target, emb_target = utils.get_vocab(args.VOCAB_TGT)

    train_source, train_target, train_s_t, target_oovs, train_offset = utils.load_dataset(args.DATA, vocab_source, vocab_target)

    assert len(train_source) == len(train_target)
    train_data = [(s, t, st, v, o) for s, t, st, v, o in six.moves.zip(
        train_source, train_target, train_s_t, target_oovs, train_offset)
                  if args.min_source_sentence <= len(s) <= args.max_source_sentence and
                  args.min_source_sentence <= len(t) <= args.max_source_sentence]


    model = getattr(net, args.modelname)(
        n_vocab_source=len(vocab_source),
        n_vocab_target=len(vocab_target),
        n_emb_source=args.n_emb_source,
        n_emb_target=args.n_emb_target,
        n_emb_label=args.n_emb_label,
        n_encoder_layers=args.encoder_layer,
        n_encoder_units=args.encoder_unit,
        n_encoder_dropout=args.encoder_dropout,
        n_decoder_units=args.decoder_unit,
        emb_source=emb_source,
        emb_target=None
    )

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=utils.convert,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(
        extensions.LogReport(trigger=(1, 'epoch'))
    )
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'main/perp', 'validation/main/perp', 'elapsed_time']
        ),
        trigger=(1, 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(model, filename='model_epoch_{.updater.epoch}.npz'),
        trigger=(10, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=1))


    if args.val:
        test_source, test_target, test_s_t, test_oovs, test_offset \
            = utils.load_dataset(args.val,
                                       vocab_source,
                                       vocab_target)
        assert len(test_source) == len(test_target)
        test_data = list(
            six.moves.zip(test_source, test_target, test_s_t, test_oovs, test_offset)
        )
        test_data = [(s, t, st, v, o) for s, t, st, v, o in test_data
                     if args.min_source_sentence <= len(s) <= args.max_source_sentence and
                     args.min_source_sentence <= len(t) <= args.max_source_sentence]

        eval_model = model.copy()
        val_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(val_iter, eval_model,
                                            converter=utils.convert,
                                            device=args.gpu))
        record_trigger = training.triggers.MinValueTrigger('validation/main/perp', (10, 'epoch'))
        trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    current = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current, args.out, 'best_model.npz')
    model_setup = args.__dict__.copy()
    model_setup['model_path'] = model_path
    model_setup['datetime'] = current_datetime

    model_setup['vocab_src_path'] = os.path.join(args.out, 'vocab.src.json')
    with open(os.path.join(args.out, 'vocab.src.json'), 'w') as f:
        json.dump(vocab_source, f, ensure_ascii=False)

    model_setup['vocab_tgt_path'] = os.path.join(args.out, 'vocab.tgt.json')
    with open(os.path.join(args.out, 'vocab.tgt.json'), 'w') as f:
        json.dump(vocab_target, f, ensure_ascii=False)

    with open(os.path.join(args.out, 'setting.json'), 'w') as f:
        json.dump(model_setup, f, ensure_ascii=False)

    trainer.run()

if __name__ == '__main__':
    main()

import numpy

from chainer import cuda
import chainer
from tqdm import tqdm

import pandas as pd

# speical symbols
PAD = -1
UNK = 0
EOS = 1
BOS = 2


def get_subsequence_before_eos(seq, eos=EOS):
    index = numpy.argwhere(seq == eos)
    return seq[:index[0, 0] + 1] if len(index) > 0 else seq


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    x_seqs, y_seqs, x_y_seqs, oovs, offset = zip(*batch)
    return (to_device_batch(x_seqs), offset), to_device_batch(y_seqs), to_device_batch(x_y_seqs), oovs


def convert_with_ret(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    x_seqs, y_seqs, ret_x_seqs, ret_y_seqs, x_y_seqs, oovs, offset, ret_offset = zip(*batch)
    return (to_device_batch(x_seqs), offset, to_device_batch(ret_x_seqs), ret_offset, to_device_batch(ret_y_seqs)), to_device_batch(y_seqs), to_device_batch(x_y_seqs), oovs


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def calculate_unknown_ratio(data, unk_threshold):
    unknown = sum((s >= unk_threshold).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total

def article2ids(words, vocab_src, vocab_tgt, oov=None):
    ids = []
    x_in_y_ids = []
    if oov is None:
        oov = []
    else:
        oov = [w for w in oov]
    for w in words:
        if w.isdigit():
            w = '<num>'
        i = vocab_src.get(w, UNK)
        ids.append(i)
        if w in vocab_tgt:
            wid = vocab_tgt[w]
        else:
            if w not in oov:
                oov.append(w)
            wid = len(vocab_tgt) + oov.index(w)
        x_in_y_ids.append(wid)
    return ids, x_in_y_ids, oov

def abstract2ids(words, vocab_tgt, oov):
    ids = []
    for w in words:
        if w.isdigit():
            w = '<num>'
        i = vocab_tgt.get(w, UNK)
        if i == UNK:
            if w in oov:
                vocab_idx = len(vocab_tgt) + oov.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(UNK)
        else:
            ids.append(i)
    return ids


def load_dataset(src_path, src_vocab, tgt_vocab):
    with open(src_path, 'r') as fi:
        src_data = []
        tgt_data = []
        xs_in_ys_data = []
        oov_data = []
        offest_data = []
        for line in fi:
            c_id, src, offest, tgt = line.split('\t')
            if len(offest.split(' ')) == 2:
                si, ei = offest.split(' ')
                si, ei = int(si) + 1, int(ei) + 2
                offest_data.append((si, ei))
            elif len(offest.split(' ')) == 3:
                si, ei, label = offest.split(' ')
                si, ei = int(si) + 1, int(ei) + 2
                offest_data.append((si, ei))
            # source
            words = src.strip().split(' ')
            art_words, xs_in_ys, oov = article2ids(words, src_vocab, tgt_vocab)
            array = numpy.array(art_words)
            src_data.append(array)
            array = numpy.array(xs_in_ys)
            xs_in_ys_data.append(array)
            # target
            words = tgt.strip().split(' ')
            abs_words = abstract2ids(words, tgt_vocab, oov)
            array = numpy.array(abs_words)
            tgt_data.append(array)
            oov_data.append(oov)
        return src_data, tgt_data, xs_in_ys_data, oov_data, offest_data


def load_dataset_with_ret(src_path, src_vocab, tgt_vocab, ret_rank=0, copy_source=[0, 1, 2]):
    df = pd.read_csv(src_path, encoding='utf-16', sep='\t', index_col=0)

    src_data = []
    tgt_data = []
    xs_in_ys_data = []
    oov_data = []
    offest_data = []
    ret_src_data = []
    ret_tgt_data = []
    ret_offest_data = []

    cols = ['input', 'offset', 'gold', 'ret input {}'.format(ret_rank), 'ret offset {}'.format(ret_rank), 'output {}'.format(ret_rank)]
    for src, offest, tgt, ret_src, ret_offset, ret_tgt in zip(*[df[col] for col in cols]):
        oov = []
        # ret target
        words = ret_tgt.strip().split(' ')
        if 2 in copy_source:
            art_words, ret_tgt_xs_in_ys, oov = article2ids(words, tgt_vocab, tgt_vocab, oov)
        else:
            art_words, ret_tgt_xs_in_ys, _ = article2ids(words, tgt_vocab, tgt_vocab)
        array = numpy.array(art_words)
        ret_tgt_data.append(array)

        si, ei = offest.split(' ')
        si, ei = int(si) + 1, int(ei) + 2
        offest_data.append((si, ei))

        # source
        words = src.strip().split(' ')
        if 0 in copy_source:
            art_words, src_xs_in_ys, oov = article2ids(words, src_vocab, tgt_vocab, oov)
        else:
            art_words, src_xs_in_ys, _ = article2ids(words, src_vocab, tgt_vocab)
        array = numpy.array(art_words)
        src_data.append(array)

        si, ei = ret_offset.split(' ')
        si, ei = int(si) + 1, int(ei) + 2
        ret_offest_data.append((si, ei))

        # ret source
        words = ret_src.strip().split(' ')
        if 1 in copy_source:
            art_words, ret_src_xs_in_ys, oov = article2ids(words, src_vocab, tgt_vocab, oov)
        else:
            art_words, ret_src_xs_in_ys, _ = article2ids(words, src_vocab, tgt_vocab)
        array = numpy.array(art_words)
        ret_src_data.append(array)

        # target
        words = tgt.strip().split(' ')
        target_words = words
        abs_words = abstract2ids(words, tgt_vocab, oov)
        array = numpy.array(abs_words)
        tgt_data.append(array)
        oov_data.append(oov)
        xs_in_ys_ls = []
        for i in copy_source:
            xs_in_ys_ls.extend([src_xs_in_ys, ret_src_xs_in_ys, ret_tgt_xs_in_ys][i])
        array = numpy.array(xs_in_ys_ls)
        xs_in_ys_data.append(array)

    return src_data, tgt_data, ret_src_data, ret_tgt_data, xs_in_ys_data, oov_data, offest_data, ret_offest_data


def get_vocab(path, unk=True, eos=False):
    vocab_dic = {}
    emb_w = []

    with open(path) as f:
        line = f.readline().split(' ')
        if len(line) == 2:
            n_lines = int(line[0])
            is_skip = True
        else:
            n_lines = 0
            is_skip = False

    with open(path) as f:
        if is_skip:
            f.readline()
        idx = 3
        tmp_dic = {'<eos>': EOS, '<bos>': BOS, '<unk>': UNK, '</s>': EOS, '<END>': EOS, '<BEGIN>': EOS}
        for i, line in tqdm(enumerate(f), total=n_lines):
            w, vec_s = line.rstrip().split(' ', 1)
            _w = w
            if w == '</s>' or w == '<END>':
                _w = '<eos>'
            if w == '<BEGIN>':
                _w = '<bos>'
            if _w in tmp_dic:
                w_i = tmp_dic[_w]
            else:
                w_i = idx
                idx += 1
            vec = numpy.array([float(v) for v in vec_s.split(' ')])
            vocab_dic[_w] = w_i
            emb_w.append(vec)
        assert list(sorted(vocab_dic.values())) == list(range(len(vocab_dic)))
        return vocab_dic, numpy.vstack(emb_w)

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

from utils import PAD, UNK, EOS

import numpy as np
from itertools import chain
from heapq import nlargest


def replace_unknown_tokens_with_unk_id(array, n_vocab):
    ret = array.copy()
    ret[ret >= n_vocab] = UNK
    return ret

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs, (ex, x_section)

def cross_entropy(ys, ts, reduce='mean', ignore_label=None, eps=1e-10):
    if isinstance(ts, Variable):
        ts = ts.data

    loss = -F.log(F.select_item(ys, ts) + eps)
    if ignore_label is not None:
        in_use = (ts != ignore_label).astype(ys.dtype)
        loss = loss * in_use
    if reduce == 'mean':
        loss = F.mean(loss)
    return loss


class EncoderDecoder(chainer.Chain):

    def __call__(self, input, ys, xs_in_ys, oovs):

        def add_eos_s(ws):
            return self.xp.hstack((ws, self.xp.array([EOS])))

        def add_eos_p(ws):
            return self.xp.hstack((self.xp.array([EOS]), ws))
        input_ys = [add_eos_p(y) for y in ys]
        gold_ys = [add_eos_s(y) for y in ys]

        agenda, hxs = self.encode(*input)
        assert all(len(a) == b.shape[0] for a, b in zip(xs_in_ys, hxs))
        agenda_reshaped = agenda.reshape((1, agenda.shape[0], agenda.shape[1]))

        hx, cx, dist, p_gen = self.decoder(agenda_reshaped, None, input_ys, xs_in_ys, hxs)

        concatenated_ys = F.hstack(gold_ys)
        n_words = concatenated_ys.shape[0]

        words_loss = cross_entropy(
            dist, concatenated_ys, reduce='no', ignore_label=PAD
        )
        loss = F.sum(words_loss)
        loss = loss / n_words
        chainer.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data)
        chainer.report({'perp': perp}, self)

        return loss

    def predict_multiple(self, input, word_xs_in_ys, max_length=50, beam_width=5):
        count = 0
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            agenda, hys = self.encode(*input)
            # agenda_reshaped = agenda.reshape((1, agenda.shape[0], agenda.shape[1]))

            heaps = [[] for _ in range(max_length + 1)]
            for _agenda, _word_xs_in_ys, _hys in zip(agenda, word_xs_in_ys, hys):
                agenda_reshaped = _agenda.reshape((1, 1, _agenda.shape[0]))
                heaps[0].append([(0, [EOS], [], [], (agenda_reshaped, None, _word_xs_in_ys, _hys))])

            outputs = [[] for _ in range(len(hys))]

            for i in range(max_length):
                cur_items_all = []
                states_all = []
                next_wids_all = []
                ids_all = []
                for j, i_items in enumerate(heaps[i]):
                    cur_items = nlargest(beam_width, i_items, key=lambda t: t[0])
                    states = [hoge[4] for hoge in cur_items]
                    next_wids = [[ci[1][-1]] for ci in cur_items]
                    cur_items_all.extend(cur_items)
                    states_all.extend(states)
                    next_wids_all.extend(next_wids)
                    ids_all.extend([j for _ in range(len(cur_items))])
                    heaps[i + 1].append([])
                if len(states_all) == 0:
                    break
                next_wids_all = self.xp.array(next_wids_all, 'i')
                new_states, wys, p_gens = self.decoder.predict_next(states_all, next_wids_all)
                wys = chainer.cuda.to_cpu(wys.data)
                if p_gens is None:
                    p_gens = np.zeros((wys.shape[0], 1))
                else:
                    p_gens = chainer.cuda.to_cpu(p_gens.data)
                p_gens = p_gens[:, 0]

                for (score, words, probs, all_p_gens, state), wy, p_gen, new_state, data_id in zip(cur_items_all, wys, p_gens, new_states, ids_all):
                    wid = words[-1]
                    if i > 0 and wid == EOS:
                        outputs[data_id].append((score, words, probs, all_p_gens))
                    else:
                        wy[UNK] = -9999
                        for next_wid in np.argsort(wy)[::-1][:beam_width]:
                            next_wid = int(next_wid)

                            next_score = score + np.log(wy[next_wid] + 1e-30)
                            count += 1
                            next_words = words + [next_wid]
                            next_probs = probs + [float(wy[next_wid])]
                            next_p_gens = all_p_gens + [float(p_gen)]
                            next_item = (next_score, next_words, next_probs, next_p_gens, new_state)

                            heaps[i + 1][data_id].append(next_item)

        for data_id, items in enumerate(heaps[-1]):
            for score, words, probs, p_gens, state in items:
                outputs[data_id].append((score, words, probs, p_gens))
        return outputs


# Retrieval-based
class Seq2Seq(EncoderDecoder):

    def __init__(self, **kwargs):
        n_vocab_source = kwargs['n_vocab_source']
        n_vocab_target = kwargs['n_vocab_target']
        n_emb_source = kwargs['n_emb_source']
        n_emb_target = kwargs['n_emb_target']
        n_encoder_layers = kwargs['n_encoder_layers']
        n_encoder_units = kwargs['n_encoder_units']
        n_encoder_dropout = kwargs['n_encoder_dropout']
        n_decoder_units = kwargs['n_decoder_units']
        emb_source = kwargs.get('emb_source', None)
        emb_target = kwargs.get('emb_target', None)
        self.weight_edit_loss = kwargs.get('weight_edit_loss', 0)
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab_source, n_emb_source, initialW=emb_source, ignore_label=PAD)
            self.embed_y = L.EmbedID(n_vocab_target, n_emb_target, initialW=emb_target, ignore_label=PAD)
            self.decoder = VanillaDecoder(
                n_vocab_target,
                n_emb_target,
                n_decoder_units,
                n_encoder_units * 2,  # because of bi-directional lstm
                self.embed_y
            )
            self.encoder = SourceEncoder(
                n_vocab_source,
                n_encoder_layers,
                n_emb_source,
                n_encoder_units,
                n_encoder_dropout,
                self.embed_x
            )
            self.W = L.Linear(n_encoder_units * 2, n_decoder_units)

    def encode(self, xs, offset):
        agenda, hs = self.encoder(xs, offset)
        return self.W(agenda), hs


# Simple Generation
class PointerGenerator(EncoderDecoder):

    def __init__(self, **kwargs):
        n_vocab_source = kwargs['n_vocab_source']
        n_vocab_target = kwargs['n_vocab_target']
        n_emb_source = kwargs['n_emb_source']
        n_emb_target = kwargs['n_emb_target']
        n_encoder_layers =  kwargs['n_encoder_layers']
        n_encoder_units = kwargs['n_encoder_units']
        n_encoder_dropout = kwargs['n_encoder_dropout']
        n_decoder_units = kwargs['n_decoder_units']
        emb_source = kwargs.get('emb_source', None)
        emb_target = kwargs.get('emb_target', None)
        self.weight_edit_loss = kwargs.get('weight_edit_loss', 0)
        super(PointerGenerator, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab_source, n_emb_source, initialW=emb_source, ignore_label=PAD)
            self.embed_y = L.EmbedID(n_vocab_target, n_emb_target, initialW=emb_target, ignore_label=PAD)
            self.encoder = SourceEncoder(
                n_vocab_source,
                n_encoder_layers,
                n_emb_source,
                n_encoder_units,
                n_encoder_dropout,
                self.embed_x
            )
            self.decoder = PGDecoder(
                n_vocab_target,
                n_emb_target,
                n_decoder_units,
                n_encoder_units * 2,  # because of bi-directional lstm
                self.embed_y
            )
            self.W = L.Linear(n_encoder_units * 2, n_decoder_units)

    def encode(self, xs, offset):
        agenda, hs = self.encoder(xs, offset)
        return self.W(agenda), hs


# Retrieve-and-edit
class PointerGeneratorWithRet(EncoderDecoder):

    def __init__(self, **kwargs):
        n_vocab_source = kwargs['n_vocab_source']
        n_vocab_target = kwargs['n_vocab_target']
        n_emb_source = kwargs['n_emb_source']
        n_emb_target = kwargs['n_emb_target']
        n_encoder_layers = kwargs['n_encoder_layers']
        n_encoder_units = kwargs['n_encoder_units']
        n_encoder_dropout = kwargs['n_encoder_dropout']
        n_decoder_units = kwargs['n_decoder_units']
        emb_source = kwargs.get('emb_source', None)
        emb_target = kwargs.get('emb_target', None)
        self.copy_source = kwargs['copy_source']
        self.weight_edit_loss = kwargs.get('weight_edit_loss', 0)
        super(PointerGeneratorWithRet, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab_source, n_emb_source, initialW=emb_source, ignore_label=PAD)
            self.embed_y = L.EmbedID(n_vocab_target, n_emb_target, initialW=emb_target, ignore_label=PAD)
            if self.weight_edit_loss > 0:
                self.es_predictor = L.Linear(n_encoder_units, 1)
            self.src_encoder = SourceEncoder(
                n_vocab_source,
                n_encoder_layers,
                n_emb_source,
                n_encoder_units,
                n_encoder_dropout,
                self.embed_x
            )
            self.ret_src_encoder = SourceEncoder(
                n_vocab_source,
                n_encoder_layers,
                n_emb_source,
                n_encoder_units,
                n_encoder_dropout,
                self.embed_x
            )
            self.ret_com_encoder = CommentEncoder(
                n_vocab_source,
                n_encoder_layers,
                n_emb_target,
                n_encoder_units,
                n_encoder_dropout,
                self.embed_y
            )
            self.decoder = PGDecoder(
                n_vocab_target,
                n_emb_target,
                n_decoder_units,
                n_encoder_units * 2,  # because of bi-directional lstm
                self.embed_y
            )
            self.W = L.Linear(n_encoder_units * 2 * 3, n_decoder_units)

    def encode(self, xs, offset, ret_xs, ret_offset, ret_ys):
        xs_agenda, xs_hs = self.src_encoder(xs, offset)
        ret_xs_agenda, ret_xs_hs = self.ret_src_encoder(ret_xs, ret_offset)
        ret_ys_agenda, ret_ys_hs = self.ret_com_encoder(ret_ys)

        agenda = self.W(F.concat([xs_agenda, ret_xs_agenda, ret_ys_agenda]))
        assert len(xs_hs) == len(ret_xs_hs) == len(ret_ys_hs)
        def concat_copy_source_hs(src_hs, ret_src_hs, ret_tgt_hs):
            ret_hs = []
            for i in self.copy_source:
                ret_hs.append([src_hs, ret_src_hs, ret_tgt_hs][i])
            return F.vstack(ret_hs)
        hs = [concat_copy_source_hs(xs_hs[i], ret_xs_hs[i], ret_ys_hs[i]) for i in range(len(xs_hs))]
        return agenda, hs


class SourceEncoder(chainer.Chain):

    def __init__(self, n_vocab, n_layers, n_emb, n_units, dropout, embed_xy):
        super(SourceEncoder, self).__init__()
        with self.init_scope():
            self.embed_x = embed_xy
            self.bilstm = L.NStepBiLSTM(
                n_layers,
                n_emb,
                n_units,
                dropout
            )
        self.n_vocab = n_vocab

    def __call__(self, xs, offset):
        errors = [self.xp.zeros((len(x), ), 'i') for x in xs]
        for err, o in zip(errors, offset):
            err[o[0]:o[1]] = 1
        exs_w, _ = sequence_embed(self.embed_x, xs)
        hy, cy, ys = self.bilstm(None, None, exs_w)
        agenda = F.vstack([F.average(y[o[0]:o[1]], axis=0) for o, y in zip(offset, ys)])
        return agenda, ys

class CommentEncoder(chainer.Chain):

    def __init__(self, n_vocab, n_layers, n_emb, n_units, dropout, embed_xy):
        super(CommentEncoder, self).__init__()
        with self.init_scope():
            self.embed_x = embed_xy
            self.bilstm = L.NStepBiLSTM(
                n_layers,
                n_emb,
                n_units,
                dropout
            )
        self.n_vocab = n_vocab

    def __call__(self, xs):
        exs, _ = sequence_embed(self.embed_x, xs)
        hy, cy, ys = self.bilstm(None, None, exs)
        agenda = F.concat(hy)
        return agenda, ys


class Decoder(chainer.Chain):

    def predict_next(self, states, ys):
        state = (F.concat([s[0] for s in states]),
                 F.concat([s[1] for s in states]) if states[0][1] is not None else None,
                 [s[2] for s in states],
                 [s[3] for s in states])

        h, c, xs_in_ys, hxs = state
        h, c, dist, p_gen = self.__call__(h, c, ys, xs_in_ys, hxs)

        new_states = [(h[:, i:i + 1], c[:, i:i + 1], xs_in_ys[i], hxs[i]) for i in range(len(states))]
        return new_states, dist, p_gen


class VanillaDecoder(Decoder):

    def __init__(self, n_vocab, n_emb, n_units, n_encoder_output_units, embed_xy):
        super(VanillaDecoder, self).__init__()
        with self.init_scope():
            self.embed_y = embed_xy
            self.lstm = L.NStepLSTM(1, n_emb, n_units, 0.3)
            self.W_s = L.Linear(n_units, n_vocab)
        self.n_vocab = n_vocab
        self.n_units = n_units

    def __call__(self, h, c, ys, xs_in_ys, hxs):

        ys_rep = [replace_unknown_tokens_with_unk_id(y, self.n_vocab) for y in ys]
        eys, _ = sequence_embed(self.embed_y, ys_rep)
        hx, cx, hys = self.lstm(h, c, eys)
        ys_size = int(max(chain([self.n_vocab - 1], *xs_in_ys))) + 1

        max_ys = max(x.shape[0] for x in hys)
        ys_tensor = F.pad_sequence(hys, max_ys, 0)

        vocab_dist_tensor = F.softmax(self.W_s(ys_tensor, n_batch_axes=2), axis=2)
        concat_vocab_dist = F.concat([_h[:y.shape[0]] for y, _h in zip(ys, vocab_dist_tensor)], axis=0)
        if ys_size - concat_vocab_dist.shape[1] > 0:
            concat_vocab_dist = F.concat([concat_vocab_dist, self.xp.zeros((concat_vocab_dist.shape[0], ys_size - concat_vocab_dist.shape[1]), 'f')])

        return hx, cx, concat_vocab_dist, None


class PGDecoder(Decoder):

    def __init__(self, n_vocab, n_emb, n_units, n_encoder_output_units, embed_xy):
        super(PGDecoder, self).__init__()
        with self.init_scope():
            self.embed_y = embed_xy
            self.lstm = L.NStepLSTM(1, n_emb, n_units, 0)

            self.W_s = L.Linear(n_units, n_vocab)
            self.W_c = L.Linear(n_units + n_encoder_output_units, n_units)
            self.W_att_h = L.Linear(n_encoder_output_units, n_units)
            self.W_att_s = L.Linear(n_units, n_units)
            self.v_att = L.Linear(n_units, 1)
            self.w_h = L.Linear(n_units, n_units)
            self.w_s = L.Linear(n_encoder_output_units, n_units)
            self.w_x = L.Linear(n_emb, n_units)
            self.w_p_gen = L.Linear(n_units, 1)
        self.n_vocab = n_vocab
        self.n_units = n_units

    def __call__(self, h, c, ys, xs_in_ys, hxs):

        ys_rep = [replace_unknown_tokens_with_unk_id(y, self.n_vocab) for y in ys]
        eys, _ = sequence_embed(self.embed_y, ys_rep)
        hx, cx, hys = self.lstm(h, c, eys)
        ys_size = int(max(chain([self.n_vocab - 1], *xs_in_ys))) + 1

        inputs = hxs
        max_eys = max(x.shape[0] for x in eys)
        eys_tensor = F.pad_sequence(eys, max_eys, 0)
        max_ys = max(x.shape[0] for x in hys)
        ys_tensor = F.pad_sequence(hys, max_ys, 0)
        max_inputs = max(x.shape[0] for x in inputs)
        inputs_tensor = F.pad_sequence(inputs, max_inputs, 0)
        inputs_mask = self.xp.vstack(
            [self.xp.hstack((self.xp.ones(x.shape[0], dtype=self.xp.float32),
                             self.xp.zeros(max_inputs - x.shape[0], dtype=self.xp.float32))) for
             x in inputs])

        inputs_mask4att = F.tile(F.expand_dims(inputs_mask * 10, 1), (1, max_ys, 1))
        att_elem_input = F.swapaxes(F.tile(F.expand_dims(self.W_att_h(inputs_tensor, n_batch_axes=2), 2),
                                (1, 1, ys_tensor.shape[1], 1)), 1, 2)
        att_elem_ys = F.tile(F.expand_dims(self.W_att_s(ys_tensor, n_batch_axes=2), 2),
                                (1, 1, inputs_tensor.shape[1], 1))
        att_logit_tensor = F.squeeze(self.v_att(F.tanh(att_elem_input + att_elem_ys), n_batch_axes=3), axis=3)
        att_tensor = F.softmax(att_logit_tensor + inputs_mask4att, axis=2)
        context_tensor = F.matmul(att_tensor, inputs_tensor)
        final_hidden = F.tanh(self.W_c(F.concat((ys_tensor, context_tensor), axis=2), n_batch_axes=2))

        onehot_list = []
        for i in range(len(ys)):
            onehot = self.xp.zeros((hxs[i].shape[0], ys_size), 'f')
            for j in range(xs_in_ys[i].shape[0]):
                onehot[j, xs_in_ys[i][j]] = 1
            onehot_list.append(onehot)
        onehot_tensor = F.pad_sequence(onehot_list, max(x.shape[0] for x in onehot_list), 0)
        source_dist_tensor = F.matmul(att_tensor, onehot_tensor)
        p_gen = F.sigmoid(self.w_p_gen(F.tanh(
            self.w_h(ys_tensor, n_batch_axes=2) + self.w_s(context_tensor, n_batch_axes=2) + self.w_x(eys_tensor, n_batch_axes=2)), n_batch_axes=2))

        concat_p_gen = F.concat([p[:y.shape[0]] for y, p in zip(ys, p_gen)], axis=0)

        weighted_source_dist_tensor = source_dist_tensor * (1 - p_gen)
        concat_source_dist = F.concat([_h[:y.shape[0]] for y, _h in zip(ys, weighted_source_dist_tensor)], axis=0)

        vocab_dist_tensor = F.softmax(self.W_s(final_hidden, n_batch_axes=2), axis=2)
        weighted_vocab_dist_tensor = vocab_dist_tensor * p_gen
        concat_vocab_dist = F.concat([_h[:y.shape[0]] for y, _h in zip(ys, weighted_vocab_dist_tensor)], axis=0)
        if ys_size - concat_vocab_dist.shape[1] > 0:
            concat_vocab_dist = F.concat([concat_vocab_dist, self.xp.zeros((concat_vocab_dist.shape[0], ys_size - concat_vocab_dist.shape[1]), 'f')])

        concat_final_dist = concat_source_dist + concat_vocab_dist

        return hx, cx, concat_final_dist, concat_p_gen

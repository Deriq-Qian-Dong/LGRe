import copy
import math
import torch
import modeling_util
import torch.nn as nn
import pytorch_pretrained_bert
from pytools import memoize_method


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)
        self.return_rep = False
        self.point = True
        self.pair = True

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape  # [4, 20]
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF  # 512-20-3

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok.cuda(), query_mask.cuda(), doc_tok.cuda(),
                                                          doc_mask.cuda())
        if self.return_rep:
            return self.cls(self.dropout(cls_reps[-1])), cls_reps[-1]
        else:
            return self.cls(self.dropout(cls_reps[-1]))


class GNNRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh().to('cuda:1')
        self.dropout = nn.Dropout(0.1).to('cuda:1')
        self.linears = clones(nn.Linear(768, 768).to('cuda:1'), 13)
        self.update_linear_a = nn.Linear(768, 768).to('cuda:1')
        self.reset_linear_a = nn.Linear(768, 768).to('cuda:1')
        self.update_linear_x = nn.Linear(768, 768).to('cuda:1')
        self.reset_linear_x = nn.Linear(768, 768).to('cuda:1')
        self.h_linear_1 = nn.Linear(768, 768).to('cuda:1')
        self.h_linear_2 = nn.Linear(768, 768).to('cuda:1')
        self.att_layer = nn.Linear(768, 1).to('cuda:1')
        self.emb_layer = nn.Linear(768, 768).to('cuda:1')
        self.output_layers = clones(nn.Linear(768 * 2, 1).to('cuda:1'), 13)
        self.gnn_cls = torch.nn.Linear(13, 1).to('cuda:1')
        self.leakyrelu = nn.LeakyReLU(0.2).to("cuda:1")
        # self.doc_att = nn.Linear(800, 1).to('cuda:1')
        # self.que_att = nn.Linear(20, 1).to('cuda:1')
        self.bert.to('cuda:0')

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape  # [4, 20]
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF  # 512-20-3

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query doc
        query_doc_results = []
        for layer in result:
            query_doc_result = []
            for i in range(layer.shape[0] // BATCH):
                query_doc_result.append(layer[i * BATCH:(i + 1) * BATCH])
            query_doc_result = torch.cat(query_doc_result, dim=1)
            query_doc_results.append(query_doc_result)
        masks = []
        for i in range(mask.shape[0] // BATCH):
            masks.append(mask[i * BATCH:(i + 1) * BATCH])
        masks = torch.cat(masks, dim=1)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)
        query_doc_results = torch.stack(query_doc_results)
        cls_results = torch.stack(cls_results)
        return cls_results, query_doc_results, masks, doc_results[-1], query_results[-1]

    def gru_unit(self, adj, x, mask):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.update_linear_a(a)
        z1 = self.update_linear_x(x)
        z = torch.sigmoid(z0 + z1)

        # reset gate
        r0 = self.reset_linear_a(a)
        r1 = self.reset_linear_x(x)
        r = torch.sigmoid(r0 + r1)

        # update embeddings
        h0 = self.h_linear_1(a)
        h1 = self.h_linear_2(r * x)
        h = self.act(mask * (h0 + h1))
        return h * z + x * (1 - z)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_doc_results, masks, doc_results, query_results = self.encode_bert(query_tok.to('cuda:0'),
                                                                                          query_mask.to('cuda:0'),
                                                                                          doc_tok.to('cuda:0'),
                                                                                          doc_mask.to('cuda:0'))
        # build attention adj graph
        cls_reps, query_doc_results, masks, doc_results, query_results \
            = cls_reps.to('cuda:1'), query_doc_results.to('cuda:1'), masks.to('cuda:1'), \
              doc_results.to('cuda:1'), query_results.to('cuda:1')
        query_doc = [l(query_doc_results[i]) for i, l in enumerate(self.linears)]

        query_doc = torch.stack(query_doc)
        d_k = query_doc.size(-1)
        scores = (torch.matmul(query_doc, query_doc.transpose(-2, -1)) / math.sqrt(d_k))
        extended_attention_mask = masks.unsqueeze(0).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        scores = scores + extended_attention_mask
        term_mask = torch.ones_like(scores)  # 13, 4, 846, 846
        q_len = query_results.shape[1] + 1
        d_len = doc_results.shape[1] + 2
        document_mask = torch.zeros(d_len, d_len)
        index = (torch.LongTensor(list(range(1, d_len))), torch.LongTensor(list(range(d_len - 1))))
        document_mask[index] = 1
        index = (torch.LongTensor(list(range(d_len - 1))), torch.LongTensor(list(range(1, d_len))))
        document_mask[index] = 1
        index = (torch.LongTensor(list(range(d_len))), torch.LongTensor(list(range(d_len))))
        document_mask[index] = 1
        document_mask = document_mask.to('cuda:1')
        term_mask[:, :, :q_len, :q_len] = torch.eye(q_len)
        term_mask[:, :, q_len:q_len + d_len, q_len:q_len + d_len] = document_mask
        term_mask[:, :, q_len + d_len:2 * q_len + d_len, q_len + d_len:2 * q_len + d_len] = torch.eye(q_len)
        term_mask[:, :, 2 * q_len + d_len:, 2 * q_len + d_len:] = document_mask
        term_mask[:, :, q_len:q_len + d_len, -d_len:] = torch.zeros_like(term_mask[:, :, q_len:q_len + d_len, -d_len:])
        term_mask[:, :, -d_len:, q_len:q_len + d_len] = torch.zeros_like(term_mask[:, :, q_len:q_len + d_len, -d_len:])
        term_mask = (1.0 - term_mask) * -10000.0
        scores = scores + term_mask
        attention_adj = self.dropout(nn.Softmax(dim=-1)(scores))  # 13, 4, 846, 846

        output = query_doc_results
        for i in range(2):
            output = self.gru_unit(attention_adj, output, masks.unsqueeze(0).unsqueeze(-1))  # 13, 4, 846, 768
        att = torch.sigmoid(self.att_layer(output))  # 13, 4, 846, 1
        emb = self.act(self.emb_layer(output))  # 13, 4, 846, 768

        g = masks.unsqueeze(0).unsqueeze(-1) * att * emb
        g = torch.sum(g, dim=-2) / masks.sum(dim=1).unsqueeze(0).unsqueeze(-1) + torch.max(g, dim=-2)[0]
        g = self.dropout(g)

        rels = torch.stack(
            [l(torch.cat([g[i], cls_reps[i]], dim=1)) for i, l in enumerate(self.output_layers)]).permute(1, 0,
                                                                                                          2).squeeze(-1)
        return self.gnn_cls(rels)


class PipelineParalleGNNRanker(GNNRanker):
    def __init__(self):
        super(PipelineParalleGNNRanker, self).__init__()
        self.split_size = 2
        self.strategy = 2

    def step1(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_doc_results, masks, doc_results, query_results = self.encode_bert(query_tok.to('cuda:0'),
                                                                                          query_mask.to('cuda:0'),
                                                                                          doc_tok.to('cuda:0'),
                                                                                          doc_mask.to('cuda:0'))
        return cls_reps, query_doc_results, masks, doc_results, query_results

    def step2(self, cls_reps, query_doc_results, masks, doc_results, query_results):
        # build attention adj graph
        cls_reps, query_doc_results, masks = cls_reps.to('cuda:1'), \
                                             query_doc_results.to('cuda:1'), \
                                             masks.to('cuda:1')
        doc_results, query_results = doc_results.to('cuda:1'), query_results.to('cuda:1')
        cos_qd = torch.cosine_similarity(
            query_results.unsqueeze(-2).expand((query_results.shape[0], query_results.shape[1],
                                                doc_results.shape[1], query_results.shape[2])),
            doc_results.unsqueeze(1).expand((doc_results.shape[0], query_results.shape[1],
                                             doc_results.shape[1], doc_results.shape[2])))
        query_doc = [l(query_doc_results[i]) for i, l in enumerate(self.linears)]
        query_doc = torch.stack(query_doc)
        d_k = query_doc.size(-1)
        scores = torch.matmul(query_doc, query_doc.transpose(-2, -1)) / math.sqrt(d_k)
        extended_attention_mask = masks.unsqueeze(0).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        scores = scores + extended_attention_mask
        if self.strategy == 1:
            term_mask = torch.ones_like(scores)  # 13, 4, 846, 846
            q_len = query_results.shape[1] + 1
            if scores.shape[-1] > 512:
                d_len = (scores.shape[-1] - 2 * q_len) // 2
            else:
                d_len = scores.shape[-1] - q_len
            term_mask[:, :, :q_len, :q_len] = torch.eye(q_len)
            term_mask[:, :, q_len:q_len + d_len, q_len:q_len + d_len] = torch.eye(d_len)
            if scores.shape[-1] > 512:
                term_mask[:, :, q_len + d_len:2 * q_len + d_len, q_len + d_len:2 * q_len + d_len] = torch.eye(q_len)
                term_mask[:, :, 2 * q_len + d_len:, 2 * q_len + d_len:] = torch.eye(d_len)
                term_mask[:, :, q_len:q_len + d_len, -d_len:] = torch.zeros_like(
                    term_mask[:, :, q_len:q_len + d_len, -d_len:])
                term_mask[:, :, -d_len:, q_len:q_len + d_len] = torch.zeros_like(
                    term_mask[:, :, q_len:q_len + d_len, -d_len:])
            term_mask = (1.0 - term_mask) * -10000.0
            scores = scores + term_mask
        if self.strategy == 2:
            term_mask = torch.ones_like(scores)  # 13, 4, 846, 846
            q_len = query_results.shape[1] + 1
            if scores.shape[-1] > 512:
                d_len = (scores.shape[-1] - 2 * q_len) // 2
            else:
                d_len = scores.shape[-1] - q_len
            document_mask = torch.zeros(d_len, d_len)
            index = (torch.LongTensor(list(range(1, d_len))), torch.LongTensor(list(range(d_len - 1))))
            document_mask[index] = 1
            index = (torch.LongTensor(list(range(d_len - 1))), torch.LongTensor(list(range(1, d_len))))
            document_mask[index] = 1
            index = (torch.LongTensor(list(range(d_len))), torch.LongTensor(list(range(d_len))))
            document_mask[index] = 1
            document_mask = document_mask.to('cuda:1')
            term_mask[:, :, :q_len, :q_len] = torch.eye(q_len)
            term_mask[:, :, q_len:q_len + d_len, q_len:q_len + d_len] = document_mask
            if scores.shape[-1] > 512:
                term_mask[:, :, q_len + d_len:2 * q_len + d_len, q_len + d_len:2 * q_len + d_len] = torch.eye(q_len)
                term_mask[:, :, 2 * q_len + d_len:, 2 * q_len + d_len:] = document_mask
                term_mask[:, :, q_len:q_len + d_len, -d_len:] = torch.zeros_like(
                    term_mask[:, :, q_len:q_len + d_len, -d_len:])
                term_mask[:, :, -d_len:, q_len:q_len + d_len] = torch.zeros_like(
                    term_mask[:, :, q_len:q_len + d_len, -d_len:])
            term_mask = (1.0 - term_mask) * -10000.0
            scores = scores + term_mask
        attention_adj = self.dropout(nn.Softmax(dim=-1)(scores))  # 13, 4, 846, 846
        output = query_doc_results
        for i in range(2):
            output = self.gru_unit(attention_adj, output, masks.unsqueeze(0).unsqueeze(-1))  # 13, 4, 846, 768
        att = torch.sigmoid(self.att_layer(output))  # 13, 4, 846, 1
        emb = self.act(self.emb_layer(output))  # 13, 4, 846, 768

        g = masks.unsqueeze(0).unsqueeze(-1) * att * emb
        g = torch.sum(g, dim=-2) / masks.sum(dim=1).unsqueeze(0).unsqueeze(-1) + torch.max(g, dim=-2)[0]
        g = self.dropout(g)

        rels = torch.stack(
            [l(torch.cat([g[i], cls_reps[i]], dim=1)) for i, l in enumerate(self.output_layers)]).permute(1, 0,
                                                                                                          2).squeeze(-1)
        return self.gnn_cls(rels), torch.cat([g, cls_reps], dim=-1).permute(1, 0, 2), \
               torch.mean(cos_qd, dim=[1, 2])

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        query_toks = iter(query_tok.split(split_size=self.split_size, dim=0))
        query_masks = iter(query_mask.split(split_size=self.split_size, dim=0))
        doc_toks = iter(doc_tok.split(split_size=self.split_size, dim=0))
        doc_masks = iter(doc_mask.split(split_size=self.split_size, dim=0))
        rets = []
        reps = []
        cos_qd = []
        cls_reps, query_doc_results, masks, doc_results, query_results = self.step1(next(query_toks), next(query_masks),
                                                                                    next(doc_toks),
                                                                                    next(doc_masks))
        for qt_next, qm_next, dt_next, dm_next in zip(query_toks, query_masks, doc_toks, doc_masks):
            ret, rep, cos = self.step2(cls_reps, query_doc_results, masks, doc_results, query_results)
            rets.append(ret)
            reps.append(rep)
            cos_qd.append(cos)
            cls_reps, query_doc_results, masks, doc_results, query_results = self.step1(qt_next, qm_next, dt_next,
                                                                                        dm_next)
        ret, rep, cos = self.step2(cls_reps, query_doc_results, masks, doc_results, query_results)
        rets.append(ret)
        reps.append(rep)
        cos_qd.append(cos)
        if self.return_rep:
            return torch.cat(rets), torch.cat(reps).reshape(-1, self.BERT_SIZE * self.CHANNELS * 2), torch.cat(cos_qd)
        else:
            return torch.cat(rets)


class CedrKnrmRanker(BertRanker):
    def __init__(self):
        super().__init__()
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        query_tok, query_mask, doc_tok, doc_mask = query_tok.cuda(), query_mask.cuda(), doc_tok.cuda(), doc_mask.cuda()
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3)  # sum over document
        mask = (simmat.sum(dim=3) != 0.)  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        scores = self.combine(result)  # linear combination over kernels
        return scores


class CustomBertModel(pytorch_pretrained_bert.BertModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=True)
        return [embedding_output] + encoded_layers

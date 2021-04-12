import os
import data
import torch
import random
import tempfile
import modeling
import argparse
import pytrec_eval
import torch.nn as nn
from tqdm import tqdm
from statistics import mean
from collections import defaultdict

SEED = 42
LR = 0.00005
BERT_LR = 1e-5
MAX_EPOCH = 20
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 32
GRAD_ACC_SIZE = 2
# other possibilities: ndcg P_20
VALIDATION_METRIC = 'P_20'
PATIENCE = 20  # how many epochs to wait for validation improvement
print(VALIDATION_METRIC)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

MODEL_MAP = {
    'GNNRanker': modeling.PipelineParalleGNNRanker,
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_knrm': modeling.CedrKnrmRanker
}


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir=None):
    '''
        Runs the training loop, controlled by the constants above
        Args:
            model(torch.nn.model or str): One of the models in modelling.py,
            or one of the keys of MODEL_MAP.
            dataset: A tuple containing two dictionaries, which contains the
            text of documents and queries in both training and validation sets:
                ({"q1" : "query text 1"}, {"d1" : "doct text 1"} )
            train_pairs: A dictionary containing query document mappings for the training set
            (i.e, document to to generate pairs from). E.g.:
                {"q1: : ["d1", "d2", "d3"]}
            qrels_train(dict): A dicationary containing training qrels. Scores > 0 are considered
            relevant. Missing scores are considered non-relevant. e.g.:
                {"q1" : {"d1" : 2, "d2" : 0}}
            If you want to generate pairs from qrels, you can pass in same object for qrels_train and train_pairs
            valid_run: Query document mappings for validation set, in same format as train_pairs.
            qrels_valid: A dictionary  containing qrels
            model_out_dir: Location where to write the models. If None, a temporary directoy is used.
    '''

    if isinstance(model, str):
        model = MODEL_MAP[model]().cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    top_valid_score = None
    print("point:", model.point)
    print("pair:", model.pair)
    top_valid_score_epoch = 0
    fg = True
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train)
        print(f'train epoch={epoch} loss={loss}')
        if fg or epoch and epoch % 5 == 0 or epoch == MAX_EPOCH - 1:
            valid_score, run_scores = validate(model, dataset, valid_run, qrels_valid, epoch)
            print(f'validation epoch={epoch} score={valid_score}')

            if top_valid_score is None or valid_score > top_valid_score:
                top_valid_score = valid_score
                print('new top validation score, saving weights', flush=True)
                model.save(os.path.join(model_out_dir, 'weights.p'))
                write_run(run_scores, model_out_dir + "/best.run")
                top_valid_score_epoch = epoch
            if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
                print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
                break


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            if model.return_rep:
                scores, reps, cos_qd = model(record['query_tok'],
                                             record['query_mask'],
                                             record['doc_tok'],
                                             record['doc_mask'])
                count = len(record['query_id']) // 2
                scores = scores.reshape(count, 2)
                cosine_similarity = torch.cosine_similarity(reps[::2], reps[1::2])
                distinguished_loss = torch.mean(1 + cosine_similarity)
                pairwise_loss = torch.mean(1. - scores.softmax(dim=1)[:, 0] + scores.softmax(dim=1)[:, 1])
                cos_qd_loss = torch.mean(1 - cos_qd[::2]) + torch.mean(1 + cos_qd[1::2])
                loss = pairwise_loss
                if model.point:
                    loss += 0.01 * pairwise_loss.item() / distinguished_loss.item() * distinguished_loss
                if model.pair:
                    loss += 0.01 * pairwise_loss.item() / cos_qd_loss.item() * cos_qd_loss
            else:
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])
                count = len(record['query_id']) // 2
                scores = scores.reshape(count, 2)
                loss = torch.mean(1. - scores.softmax(dim=1)[:, 0] + scores.softmax(dim=1)[:, 1])  # pairwise_loss
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, valid_qrels, epoch):
    run_scores = run_model(model, dataset, run)
    metric = VALIDATION_METRIC
    if metric.startswith("P_"):
        metric = "P"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
    eval_scores = trec_eval.evaluate(run_scores)
    return mean([d[VALIDATION_METRIC] for d in eval_scores.values()]), run_scores


def run_model(model, dataset, run, desc='valid'):
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE * 2):
            if model.return_rep:
                scores, _, _ = model(records['query_tok'],
                                     records['query_mask'],
                                     records['doc_tok'],
                                     records['doc_mask'])
            else:
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run


def write_run(rerank_run, runf):
    '''
        Utility method to write a file to disk. Now unused
    '''
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} Q0 {did} {i + 1} {score} run\n')


def main_cli():
    parser = argparse.ArgumentParser('LGRe training and validation')
    parser.add_argument('--fold', type=int, help='an integer for the flod')
    args = parser.parse_args()
    model = MODEL_MAP["GNNRanker"]()
    model.return_rep = True
    fold = args.fold  #1 2 3 4 5
    model.load(os.path.join("cedr-models", 'vbert-robust-f%d.p' % fold))
    dataset = data.read_datafiles([open("data/robust/queries.tsv").readlines(),
                                   open("data/robust/documents.tsv").readlines()])
    qrels = data.read_qrels_dict(open("data/robust/qrels").readlines())
    train_pairs = data.read_pairs_dict(
        open("data/robust/f%d.train.pairs" % fold).readlines())
    valid_run = data.read_run_dict(open("data/robust/f%d.valid.run" % fold).readlines())

    os.makedirs("models/gnn-cosqd-%d" % fold, exist_ok=True)
    # # we use the same qrels object for both training and validation sets

    main(model, dataset, train_pairs, qrels, valid_run, qrels, "models/gnn-cosqd-%d" % fold)
    model.load(os.path.join("models/gnn-cosqd-%d" % fold, 'weights.p'))
    run = data.read_run_dict(open("data/robust/f%d.test.run" % fold).readlines())

    run_scores = run_model(model, dataset, run, desc='rerank')
    trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {"P_20"})
    eval_scores = trec_eval.evaluate(run_scores)
    print("models/gnn-cosqd-%d:" % fold, mean([d["P_20"] for d in eval_scores.values()]))
    write_run(run_scores, "models/gnn-cosqd-%d" % fold + "/gnn.run")


if __name__ == '__main__':
    main_cli()

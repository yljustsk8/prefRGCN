# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import sys

import numpy as np

sys.path.append("..")
sys.path.append("../..")
from argparse import ArgumentParser

from netquery.utils import *
from netquery.utils import _get_perc_scores
from netquery.bio.data_utils import load_graph
from netquery.pref_data_utils import load_prefs_by_formula
from netquery.model import QueryEncoderDecoder
from netquery.pref_model import PrefRGCN

from torch import optim
from torch_geometric.utils import dense_to_sparse
from collections import defaultdict

def load_args():
    parser = ArgumentParser()
    # graph query
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./bio_data/")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iter", type=int, default=100000000)
    parser.add_argument("--max_burn_in", type=int, default=1000000)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--decoder", type=str, default="bilinear")
    parser.add_argument("--inter_decoder", type=str, default="mean")
    parser.add_argument("--opt", type=str, default="adam")
    # pref
    parser.add_argument("--emb_model_loc", type=str, default="./bio-0-128-0.010000-bilinear-mean-edge_conv")
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--pref_dec", type=str, default="RGCN")
    parser.add_argument("--pref_graph_type", type=str, default="reverse")
    parser.add_argument("--rel_num", type=int, default=4)

    return parser.parse_args()

def run_eval(args, model, prefs, iteration, logger, batch_size=128, seed=36):
    print("evaluating...")
    random.seed(seed)
    # perc
    # auc & roc
    perc_scores = defaultdict()
    auc_scores = defaultdict()
    # calculate perc
    for pref_type in prefs:
        type_perc_scores = []
        for formula in prefs[pref_type]:
            formula_prefs = prefs[pref_type][formula]
            offset = 0
            entities_level_num = len(formula_prefs[0].sampled_entities)
            while offset < len(formula_prefs):
                max_index = min(offset + batch_size, len(formula_prefs))
                batch_prefs = formula_prefs[offset:max_index]
                for level in range(entities_level_num-1):
                    higher_entity = [random.choice(formula_prefs[j].sampled_entities[level+1]) for j in range(offset, max_index)]
                    lower_entities = [n for j in range(offset, max_index) for n in formula_prefs[j].sampled_entities[level]]
                    lengths = [len(formula_prefs[j].sampled_entities[level]) for j in range(offset, max_index)]
                    current_prefs = batch_prefs + [b for i, b in enumerate(batch_prefs) for _ in range(lengths[i])]
                    edge_idx, edge_type = dense_to_sparse(torch.Tensor([formula.edge_index for _ in range(len(current_prefs))]))
                    batch = []
                    for j in range(len(current_prefs)):
                        batch += [j for _ in range(len(formula.edge_index))]
                    batch = torch.LongTensor(batch)
                    if args.cuda:
                        edge_idx = edge_idx.cuda()
                        edge_type = edge_type.cuda()
                        batch = batch.cuda()
                    batch_scores = model.forward(
                        formula=formula,
                        prefs=current_prefs,
                        edge_index=edge_idx,
                        edge_type=edge_type,
                        batch=batch,
                        targets=higher_entity+lower_entities
                    ).data.tolist()
                    type_perc_scores.extend(_get_perc_scores(batch_scores, lengths))
                offset += batch_size
        perc_scores[pref_type] = np.mean(type_perc_scores)
    # calculate auc & roc
    # for pref_type in prefs:
    #     print(pref_type)
    #     all_labels = []
    #     all_predictions = []
    #     for formula in prefs[pref_type]:
    #         formula_labels = []
    #         formula_predictions = []
    #         formula_prefs = prefs[pref_type][formula]
    #         offset = 0
    #         while offset < len(formula_prefs):
    #             max_index = min(offset + batch_size, len(formula_prefs))
    #             batch_prefs = formula_prefs[offset:max_index]
    #             length = len(batch_prefs[0].sampled_entities)
    #             current_prefs = batch_prefs * length
    #             edge_idx, edge_type = dense_to_sparse(torch.Tensor([formula.edge_index for _ in range(len(current_prefs))]))
    #             batch = []
    #             for j in range(len(current_prefs)):
    #                 batch += [j for _ in range(len(formula.edge_index))]
    #             batch = torch.LongTensor(batch)
    #             targets = [random.choice(pref.sampled_entities[level]) for level in range(length) for pref in batch_prefs]
    #             labels = [level for level in range(length) for pref in batch_prefs]
    #             batch_scores = model.forward(
    #                 formula=formula,
    #                 prefs=current_prefs,
    #                 edge_index=edge_idx,
    #                 batch=batch,
    #                 targets=targets
    #             ).data.tolist()
    #             formula_labels.extend(labels)
    #             formula_predictions.extend(batch_scores)
    #             offset += batch_size
    #
    #         all_labels.extend(formula_labels)
    #         all_predictions.extend(formula_predictions)
    #     overall_auc = roc_auc_score(all_labels, np.nan_to_num(all_predictions))
    #     auc_scores[pref_type] = overall_auc

    return perc_scores, auc_scores

def run_train():
    args = load_args()

    print("Loading graph data..")
    graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
    if args.cuda:
        graph.features = cudify(feature_modules, node_maps)
    out_dims = {mode:args.embed_dim for mode in graph.relations}


    print("Loading preference data...")
    train_prefs = defaultdict(lambda : defaultdict(list))
    test_prefs = defaultdict(lambda : defaultdict(list))

    for i in range(1, 4):
        i_train_prefs = load_prefs_by_formula(args.data_dir + "preference/train_pref_{:d}.pkl".format(i))
        i_test_prefs = load_prefs_by_formula(args.data_dir + "preference/test_pref_{:d}.pkl".format(i))
        train_prefs.update(i_train_prefs)
        test_prefs.update(i_test_prefs)

    # cudify formulas
    if args.cuda:
        for key in train_prefs:
            for formula in train_prefs[key]:
                formula.cudify()
        for key in test_prefs:
            for formula in test_prefs[key]:
                formula.cudify()
    if args.pref_graph_type == "reverse":
        for key in train_prefs:
            for formula in train_prefs[key]:
                formula.edge_index = formula.reverse_edge_idx
        for key in test_prefs:
            for formula in test_prefs[key]:
                formula.edge_index = formula.reverse_edge_idx

    enc = get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)
    dec = get_metapath_decoder(graph, enc.out_dims if args.depth > 0 else out_dims, args.decoder)
    inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder)

    enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
    # enc_dec.load_state_dict(torch.load(args.emb_model_loc))

    prefRGCN = PrefRGCN(enc_dec, args.decoder, args.embed_dim, args.hidden)


    if args.cuda:
        prefRGCN.cuda()

    optimizer = optim.Adam([p for p in prefRGCN.parameters() if p.requires_grad], lr=args.lr)

    log_file = args.log_dir + "/{data:s}-{pref_graph_type:s}-{hidden:d}-{lr:f}-{encoder:s}-{decoder:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        pref_graph_type=args.pref_graph_type,
        hidden=args.hidden,
        lr=args.lr,
        encoder="",
        decoder=args.pref_dec,
    )

    model_file = args.log_dir + "/{data:s}-{pref_graph_type:s}-{hidden:d}-{lr:f}-{encoder:s}-{decoder:s}".format(
        data=args.data_dir.strip().split("/")[-1],
        pref_graph_type=args.pref_graph_type,
        hidden=args.hidden,
        lr=args.lr,
        encoder="",
        decoder=args.pref_dec,
    )

    logger = setup_logging(log_file)
    # train
    losses = []
    ema_loss = None
    for i in range(int(10e5)):
        optimizer.zero_grad()
        loss = None

        for pref_type in train_prefs:
            if pref_type != "UIUP-1":
                continue
            # select formula and prefs
            p = train_prefs[pref_type]
            num_prefs = [float(len(prefs)) for prefs in list(p.values())]
            denom = float(sum(num_prefs))
            formula_index = np.argmax(np.random.multinomial(1, np.array(num_prefs) / denom))
            formula = list(p.keys())[formula_index]
            n = len(p[formula])
            start = (i * args.batch_size) % n
            end = min(((i + 1) * args.batch_size) % n, n)
            end = n if end <= start else end
            # calculate and update loss
            current_prefs = p[formula][start:end]
            edge_idx, edge_type = dense_to_sparse(torch.Tensor([formula.edge_index for _ in range(end-start)]))
            batch = []
            for j in range(end-start):
                batch += [j for _ in range(len(formula.edge_index))]
            batch = torch.LongTensor(batch)
            if args.cuda:
                edge_idx = edge_idx.cuda()
                edge_type = edge_type.cuda()
                batch = batch.cuda()
            if loss == None:
                loss = prefRGCN.margin_loss(formula, current_prefs, edge_idx, edge_type, batch)
            else:
                loss += prefRGCN.margin_loss(formula, current_prefs, edge_idx, edge_type, batch)

        losses.append(loss.item())
        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss = 0.99 * ema_loss + 0.01 * loss.item()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))

        if i >= args.val_every and i % args.val_every == 0:
            perc_scores, auc_scores = run_eval(args, prefRGCN, test_prefs, i, logger)
            for pref_type in perc_scores:
                logger.info("{:s} val AUC: {:f} val PERC: {:f}; iteration: {:d}".
                            format(pref_type, auc_scores[pref_type], perc_scores[pref_type], i)
                )


if __name__ == '__main__':
    run_train()
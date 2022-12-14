import torch
from collections import defaultdict
import pickle as pickle
from netquery.pref_graph import Query, Graph, SingleValPreference


def load_graph(data_dir, embed_dim):
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    node_maps = {m : {n : i for i, n in enumerate(id_list)} for m, id_list in node_maps.items()}
    for m in node_maps:
        node_maps[m][-1] = -1
    feature_dims = {m : embed_dim for m in rels}
    feature_modules = {m : torch.nn.Embedding(len(node_maps[m])+1, embed_dim) for m in rels}
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1))
    graph = Graph(features, feature_dims, rels, adj_lists)
    return graph, feature_modules, node_maps

def load_queries(data_file, keep_graph=False):
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]


def load_prefs_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    prefs = defaultdict(lambda : defaultdict(list))
    for raw_pref in raw_info:
        pref = SingleValPreference.deserialize(raw_pref, True)
        prefs[pref.formula.pref_type][pref.formula].append(pref)
    return prefs

def sample_prefs(samples, data_dir):
    train_graph, _, _ = load_graph(data_dir, 10)
    test_graph, _, _ = load_graph(data_dir, 10)
    # if test:
    print("Loading test/val data...")
    test_edges = load_queries(data_dir + "/test_edges.pkl")
    val_edges = load_queries(data_dir + "/val_edges.pkl")
    # else:
    #     test_edges = []
    #     val_edges = []
    train_graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges+val_edges])
    # if test:
    #     pref_graph = test_graph
    #     t_graph = train_graph
    # else:
    #     pref_graph = train_graph
    #     t_graph = None

    train_data = defaultdict(list)
    test_data = defaultdict(list)

    print("sampling train data...")
    for pref_type in ["UIUP-1", "UIUP-2", "UIUP-3", "UIUP-1-reverse", "UIUP-2-reverse", "UIUP-3-reverse",
                  "UICP-2", "CIUP-3a", "CIUP-3b", "CICP-3"]:
        if "1" in pref_type:
            num_pref_atts = 1
            entity_type = ["sideeffects", "function", "disease", "protein", "drug"]
        elif "2" in pref_type:
            num_pref_atts = 2
            entity_type = ["function", "disease", "protein", "drug"]
        elif "3" in pref_type:
            num_pref_atts = 3
            entity_type = ["protein", "drug"]
        current_data = train_graph.sample_singlev_preferences(
            entity_type=entity_type,
            pref_type=pref_type,
            num_samples=samples,
            num_pref_atts=num_pref_atts,
            question_sample_max=10,
            verbose=True
        )
        train_data[pref_type] = current_data
        print("finish sampling pref:", pref_type, "(train)")

    print("sampling test data...")
    for pref_type in ["UIUP-1", "UIUP-2", "UIUP-3", "UIUP-1-reverse", "UIUP-2-reverse", "UIUP-3-reverse",
                      "UICP-2", "CIUP-3a", "CIUP-3b", "CICP-3"]:
        if "1" in pref_type:
            num_pref_atts = 1
            entity_type = ["sideeffects", "function", "disease", "protein", "drug"]
        elif "2" in pref_type:
            num_pref_atts = 2
            entity_type = ["function", "disease", "protein", "drug"]
        elif "3" in pref_type:
            num_pref_atts = 3
            entity_type = ["protein", "drug"]
        current_data = test_graph.sample_singlev_preferences(
            entity_type=entity_type,
            pref_type=pref_type,
            num_samples=samples//10,
            num_pref_atts=num_pref_atts,
            question_sample_max=10,
            verbose=True
        )
        current_data = list(set(current_data) - set(train_data[pref_type]))
        test_data[pref_type] = current_data
        print("finish sampling pref:", pref_type, "(test)")

    train_pref_1 = []
    train_pref_2 = []
    train_pref_3 = []
    test_pref_1 = []
    test_pref_2 = []
    test_pref_3 = []
    for pref_type in ["UIUP-1", "UIUP-2", "UIUP-3", "UIUP-1-reverse", "UIUP-2-reverse", "UIUP-3-reverse",
                      "UICP-2", "CIUP-3a", "CIUP-3b", "CICP-3"]:
        if "1" in pref_type:
            train_pref_1 += train_data[pref_type]
            test_pref_1 += test_data[pref_type]
        if "2" in pref_type:
            train_pref_2 += train_data[pref_type]
            test_pref_2 += test_data[pref_type]
        if "3" in pref_type:
            train_pref_3 += train_data[pref_type]
            test_pref_3 += test_data[pref_type]

    pickle.dump([pref.serialize() for pref in train_pref_1], open(data_dir + "/preference/train_pref_1.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([pref.serialize() for pref in train_pref_2], open(data_dir + "/preference/train_pref_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([pref.serialize() for pref in train_pref_3], open(data_dir + "/preference/train_pref_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([pref.serialize() for pref in test_pref_1], open(data_dir + "/preference/test_pref_1.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([pref.serialize() for pref in test_pref_2], open(data_dir + "/preference/test_pref_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([pref.serialize() for pref in test_pref_3], open(data_dir + "/preference/test_pref_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == '__main__':
    data_dir = "./bio/bio_data"
    sample_prefs(samples=200000, data_dir=data_dir)

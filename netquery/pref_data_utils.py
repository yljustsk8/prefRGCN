from collections import defaultdict
import pickle as pickle
from netquery.pref_graph import SingleValPreference

def load_prefs_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    prefs = defaultdict(lambda : defaultdict(list))
    for raw_pref in raw_info:
        pref = SingleValPreference.deserialize(raw_pref, True)
        prefs[pref.formula.pref_type][pref.formula].append(pref)
    return prefs

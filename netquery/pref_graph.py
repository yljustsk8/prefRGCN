import math
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import random
import sys
import torch

def _reverse_relation(relation):
    return (relation[-1], relation[1], relation[0])


def _reverse_edge(edge):
    return (edge[-1], _reverse_relation(edge[1]), edge[0])

class Preference():

    def __init__(self, pref_info, questions, hard_questions, question_max=100, keep_info=False):
        avps = pref_info[1]
        self.type = pref_info[0]
        self.attributes = tuple([avp[0] for avp in avps])
        self.values = tuple([(avp[1], avp[2]) for avp in avps])
        self.num_atts = len(self.attributes)
        self.rel_emb = None
        self.v_emb = []
        if keep_info:
            self.pref_info = pref_info
        else:
            self.pref_info = None
        if not questions is None:
            self.questions = list(questions) if len(questions) < question_max else random.sample(questions, question_max)
        else:
            self.questions = None
        if not hard_questions is None:
            self.hard_questions = list(hard_questions) if len(hard_questions) < question_max else random.sample(hard_questions, question_max)
        else:
            self.hard_questions = None

    def __hash__(self):
        return hash((self.type, self.attributes, self.values))

    def __eq__(self, other):
        return (self.type, self.attributes, self.values) == (other.type, other.attributes, other.values)

    def __neq__(self, other):
        return (self.type, self.attributes, self.values) != (other.type, other.attributes, other.values)

    def gen_pref_graph(self):
        '''
        nodes: (num in graph, actual num)
        edge_type: 0: ordinate, 1: have, 2: att_preferred, 3: val_preferred
        edges: (head, type, tail)
        '''
        pref_graph = defaultdict(list)
        pref_graph["nodes"].append((0, "root"))
        if "UIUP" in self.type:
            # "UIUP-1", "UIUP-2", "UIUP-3"
            for i in range(self.num_atts):
                # create nodes
                pref_graph["nodes"].append((i+1, self.attributes[i]))
                pref_graph["nodes"].append((2*i+1+self.num_atts, self.values[i][0]))
                pref_graph["nodes"].append((2*i+2+self.num_atts, self.values[i][1]))
                # create edges
                pref_graph["edges"].append((0, 0, i+1))
                pref_graph["edges"].append((i+1, 1, 2*i+1+self.num_atts))
                pref_graph["edges"].append((i+1, 1, 2*i+2+self.num_atts))
                pref_graph["edges"].append((2*i+1+self.num_atts, 3, 2*i+2+self.num_atts))
                if i >= 1:
                    pref_graph["edges"].append((i, 2, i+1))
        elif self.type == "UICP-2":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[1]),
                (4, self.values[0][0]), (5, self.values[0][1]),
                (6, self.values[1][0]), (7, self.values[1][1]),
                (8, self.values[1][1]), (9, self.values[1][0]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (4, 0, 2), (5, 0, 3),
                (1, 1, 4), (1, 1, 5), (2, 1, 6), (2, 1, 7), (3, 1, 8), (3, 1, 9),
                (4, 3, 5), (6, 3, 7), (8, 3, 9)
            ]
        elif self.type == "CIUP-3a":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.values[0][0]), (5, self.values[0][1]),
                (6, self.values[1][0]), (7, self.values[1][1]),
                (8, self.values[2][0]), (9, self.values[2][1]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (4, 0, 2), (5, 0, 3),
                (1, 1, 4), (1, 1, 5), (2, 1, 6), (2, 1, 7), (3, 1, 8), (3, 1, 9),
                (4, 3, 5), (6, 3, 7), (8, 3, 9),
            ]
        elif self.type == "CIUP-3b":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.attributes[2]), (5, self.attributes[1]),
                (6, self.values[0][0]), (7, self.values[0][1]),
                (8, self.values[1][0]), (9, self.values[1][1]),
                (10, self.values[2][0]), (11, self.values[2][1]),
                (12, self.values[2][0]), (13, self.values[2][1]),
                (14, self.values[1][0]), (15, self.values[1][1]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (6, 0, 2), (6, 0, 3), (7, 0, 4), (7, 0, 5),
                (1, 1, 6), (1, 1, 7), (2, 1, 8), (2, 1, 9), (3, 1, 10), (3, 1, 11),
                (4, 1, 12), (4, 1, 13), (5, 1, 14), (5, 1, 15),
                (2, 2, 3), (4, 2, 5),
                (6, 3, 7), (8, 3, 9), (10, 3, 11), (12, 3, 13), (14, 3, 15),
            ]
        elif self.type == "CICP-3":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.attributes[2]), (5, self.attributes[1]),
                (6, self.values[0][0]), (7, self.values[0][1]),
                (8, self.values[1][0]), (9, self.values[1][1]),
                (10, self.values[2][0]), (11, self.values[2][1]),
                (12, self.values[2][1]), (13, self.values[2][0]),
                (14, self.values[1][1]), (15, self.values[1][0]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (6, 0, 2), (6, 0, 3), (7, 0, 4), (7, 0, 5),
                (1, 1, 6), (1, 1, 7), (2, 1, 8), (2, 1, 9), (3, 1, 10), (3, 1, 11),
                (4, 1, 12), (4, 1, 13), (5, 1, 14), (5, 1, 15),
                (2, 2, 3), (4, 2, 5),
                (6, 3, 7), (8, 3, 9), (10, 3, 11), (12, 3, 13), (14, 3, 15),
            ]
        return pref_graph

    def judge_pair_priority(self, e1_info, e2_info):
        # todo: judge the priority between e1 and e2
        # return: 1 means e1 > e2, 0 menas e1 < e2, -1 means e1 == e2
        e1_flags = [(self.values[i][0] in e1_info[i], self.values[i][1] in e1_info[i]) for i in range(self.num_atts)]
        e2_flags = [(self.values[i][0] in e2_info[i], self.values[i][1] in e2_info[i]) for i in range(self.num_atts)]
        judge_basis = []
        if "UIUP" in self.type:
            for idx in range(self.num_atts):
                judge_basis.append((self.attributes[idx], self.values[idx][0]))
                if e1_flags[idx][0] and e2_flags[idx][0]:
                    continue
                elif (not e1_flags[idx][0]) and e2_flags[idx][0]:
                    return 0, judge_basis
                elif (not e2_flags[idx][0]) and e1_flags[idx][0]:
                    return 1, judge_basis
                else:
                    judge_basis.append((self.attributes[idx], self.values[idx][1]))
                    if e1_flags[idx][1] and e2_flags[idx][1]:
                        continue
                    elif (not e1_flags[idx][0]) and e2_flags[idx][0]:
                        return 0, judge_basis
                    elif (not e2_flags[idx][0]) and e1_flags[idx][0]:
                        return 1, judge_basis
                    else:
                        continue
            return -1, judge_basis
        elif self.type == "UICP-2":
            judge_basis.append((self.attributes[0], self.values[0][0]))
            if e1_flags[0][0] and e2_flags[0][0]:
                judge_basis.append((self.attributes[1], self.values[1][0]))
                if e1_flags[1][0] and e2_flags[1][0]:
                    return -1, judge_basis
                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                    return 0, judge_basis
                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                    return 1, judge_basis
                else:
                    judge_basis.append((self.attributes[1], self.values[1][1]))
                    if e1_flags[1][1] and e2_flags[1][1]:
                        return -1, judge_basis
                    elif (not e1_flags[1][1]) and e2_flags[1][1]:
                        return 0, judge_basis
                    elif (not e2_flags[1][1]) and e1_flags[1][1]:
                        return 1, judge_basis
                    else:
                        return -1, judge_basis
            elif (not e1_flags[0][0]) and e2_flags[0][0]:
                return 0, judge_basis
            elif (not e2_flags[0][0]) and e1_flags[0][0]:
                return 1, judge_basis
            else:
                judge_basis.append((self.attributes[0], self.values[0][1]))
                if e1_flags[0][1] and e2_flags[0][1]:
                    judge_basis.append((self.attributes[1], self.values[1][1]))
                    if e1_flags[1][1] and e2_flags[1][1]:
                        return -1, judge_basis
                    elif (not e1_flags[1][1]) and e2_flags[1][1]:
                        return 0, judge_basis
                    elif (not e2_flags[1][1]) and e1_flags[1][1]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[1], self.values[1][0]))
                        if e1_flags[1][0] and e2_flags[1][0]:
                            return -1, judge_basis
                        elif (not e1_flags[1][0]) and e2_flags[1][0]:
                            return 0, judge_basis
                        elif (not e2_flags[1][0]) and e1_flags[1][0]:
                            return 1, judge_basis
                        else:
                            return -1, judge_basis
                elif (not e1_flags[0][1]) and e2_flags[0][1]:
                    return 0, judge_basis
                elif (not e2_flags[0][1]) and e1_flags[0][1]:
                    return 1, judge_basis
                else:
                    return -1, judge_basis
        elif self.type == "CIUP-3a":
            judge_basis.append((self.attributes[0], self.values[0][0]))
            if e1_flags[0][0] and e2_flags[0][0]:
                judge_basis.append((self.attributes[1], self.values[1][0]))
                if e1_flags[1][0] and e2_flags[1][0]:
                    return -1, judge_basis
                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                    return 0, judge_basis
                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                    return 1, judge_basis
                else:
                    judge_basis.append((self.attributes[1], self.values[1][1]))
                    if e1_flags[1][1] and e2_flags[1][1]:
                        return -1, judge_basis
                    elif (not e1_flags[1][1]) and e2_flags[1][1]:
                        return 0, judge_basis
                    elif (not e2_flags[1][1]) and e1_flags[1][1]:
                        return 1, judge_basis
                    else:
                        return -1, judge_basis
            elif (not e1_flags[0][0]) and e2_flags[0][0]:
                return 0, judge_basis
            elif (not e2_flags[0][0]) and e1_flags[0][0]:
                return 1, judge_basis
            else:
                judge_basis.append((self.attributes[0], self.values[0][1]))
                if e1_flags[0][1] and e2_flags[0][1]:
                    judge_basis.append((self.attributes[2], self.values[2][0]))
                    if e1_flags[2][0] and e2_flags[2][0]:
                        return -1, judge_basis
                    elif (not e1_flags[2][0]) and e2_flags[2][0]:
                        return 0, judge_basis
                    elif (not e2_flags[2][0]) and e1_flags[2][0]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[2], self.values[2][1]))
                        if e1_flags[2][1] and e2_flags[2][1]:
                            return -1, judge_basis
                        elif (not e1_flags[2][1]) and e2_flags[2][1]:
                            return 0, judge_basis
                        elif (not e2_flags[2][1]) and e1_flags[2][1]:
                            return 1, judge_basis
                        else:
                            return -1, judge_basis
                elif (not e1_flags[0][1]) and e2_flags[0][1]:
                    return 0, judge_basis
                elif (not e2_flags[0][1]) and e1_flags[0][1]:
                    return 1, judge_basis
                else:
                    return -1, judge_basis
        elif self.type == "CIUP-3b":
            # att0
            judge_basis.append((self.attributes[0], self.values[0][0]))
            if e1_flags[0][0] and e2_flags[0][0]:
                # att 1.0
                judge_basis.append((self.attributes[1], self.values[1][0]))
                if e1_flags[1][0] and e2_flags[1][0]:
                    # att2
                    judge_basis.append((self.attributes[2], self.values[2][0]))
                    if e1_flags[2][0] and e2_flags[2][0]:
                        return -1, judge_basis
                    elif (not e1_flags[2][0]) and e2_flags[2][0]:
                        return 0, judge_basis
                    elif (not e2_flags[2][0]) and e1_flags[2][0]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[2], self.values[2][1]))
                        if e1_flags[2][1] and e2_flags[2][1]:
                            return -1, judge_basis
                        elif (not e1_flags[2][1]) and e2_flags[2][1]:
                            return 0, judge_basis
                        elif (not e2_flags[2][1]) and e1_flags[2][1]:
                            return 1, judge_basis
                        else:
                            return -1, judge_basis
                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                    return 0, judge_basis
                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                    return 1, judge_basis
                else:
                    # att 1.1
                    judge_basis.append((self.attributes[1], self.values[1][1]))
                    if e1_flags[1][1] and e2_flags[1][1]:
                        # att2
                        judge_basis.append((self.attributes[2], self.values[2][0]))
                        if e1_flags[2][0] and e2_flags[2][0]:
                            return -1, judge_basis
                        elif (not e1_flags[2][0]) and e2_flags[2][0]:
                            return 0, judge_basis
                        elif (not e2_flags[2][0]) and e1_flags[2][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[2], self.values[2][1]))
                            if e1_flags[2][1] and e2_flags[2][1]:
                                return -1, judge_basis
                            elif (not e1_flags[2][1]) and e2_flags[2][1]:
                                return 0, judge_basis
                            elif (not e2_flags[2][1]) and e1_flags[2][1]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
                    elif (not e1_flags[1][1]) and e2_flags[1][1]:
                        return 0, judge_basis
                    elif (not e2_flags[1][1]) and e1_flags[1][1]:
                        return 1, judge_basis
                    else:
                        # att2
                        judge_basis.append((self.attributes[2], self.values[2][0]))
                        if e1_flags[2][0] and e2_flags[2][0]:
                            return -1, judge_basis
                        elif (not e1_flags[2][0]) and e2_flags[2][0]:
                            return 0, judge_basis
                        elif (not e2_flags[2][0]) and e1_flags[2][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[2], self.values[2][1]))
                            if e1_flags[2][1] and e2_flags[2][1]:
                                return -1, judge_basis
                            elif (not e1_flags[2][1]) and e2_flags[2][1]:
                                return 0, judge_basis
                            elif (not e2_flags[2][1]) and e1_flags[2][1]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
            elif (not e1_flags[0][0]) and e2_flags[0][0]:
                return 0, judge_basis
            elif (not e2_flags[0][0]) and e1_flags[0][0]:
                return 1, judge_basis
            else:
                # att0.1
                judge_basis.append((self.attributes[0], self.values[0][1]))
                if e1_flags[0][1] and e2_flags[0][1]:
                    # att 2.1
                    judge_basis.append((self.attributes[2], self.values[2][0]))
                    if e1_flags[2][0] and e2_flags[2][0]:
                        judge_basis.append((self.attributes[1], self.values[1][0]))
                        if e1_flags[1][0] and e2_flags[1][0]:
                            return -1, judge_basis
                        elif (not e1_flags[1][0]) and e2_flags[1][0]:
                            return 0, judge_basis
                        elif (not e2_flags[1][0]) and e1_flags[1][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[1], self.values[1][1]))
                            if e1_flags[1][1] and e2_flags[1][1]:
                                return -1, judge_basis
                            elif (not e1_flags[1][1]) and e2_flags[1][1]:
                                return 0, judge_basis
                            elif (not e2_flags[1][1]) and e1_flags[1][1]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
                    elif (not e1_flags[2][0]) and e2_flags[2][0]:
                        return 0, judge_basis
                    elif (not e2_flags[2][0]) and e1_flags[2][0]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[2], self.values[2][1]))
                        if e1_flags[2][1] and e2_flags[2][1]:
                            judge_basis.append((self.attributes[1], self.values[1][0]))
                            if e1_flags[1][0] and e2_flags[1][0]:
                                return -1, judge_basis
                            elif (not e1_flags[1][0]) and e2_flags[1][0]:
                                return 0, judge_basis
                            elif (not e2_flags[1][0]) and e1_flags[1][0]:
                                return 1, judge_basis
                            else:
                                judge_basis.append((self.attributes[1], self.values[1][1]))
                                if e1_flags[1][1] and e2_flags[1][1]:
                                    return -1, judge_basis
                                elif (not e1_flags[1][1]) and e2_flags[1][1]:
                                    return 0, judge_basis
                                elif (not e2_flags[1][1]) and e1_flags[1][1]:
                                    return 1, judge_basis
                                else:
                                    return -1, judge_basis
                        elif (not e1_flags[2][1]) and e2_flags[2][1]:
                            return 0, judge_basis
                        elif (not e2_flags[2][1]) and e1_flags[2][1]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[1], self.values[1][0]))
                            if e1_flags[1][0] and e2_flags[1][0]:
                                return -1, judge_basis
                            elif (not e1_flags[1][0]) and e2_flags[1][0]:
                                return 0, judge_basis
                            elif (not e2_flags[1][0]) and e1_flags[1][0]:
                                return 1, judge_basis
                            else:
                                judge_basis.append((self.attributes[1], self.values[1][1]))
                                if e1_flags[1][1] and e2_flags[1][1]:
                                    return -1, judge_basis
                                elif (not e1_flags[1][1]) and e2_flags[1][1]:
                                    return 0, judge_basis
                                elif (not e2_flags[1][1]) and e1_flags[1][1]:
                                    return 1, judge_basis
                                else:
                                    return -1, judge_basis
        else:
            # att0
            judge_basis.append((self.attributes[0], self.values[0][0]))
            if e1_flags[0][0] and e2_flags[0][0]:
                # att 1.0
                judge_basis.append((self.attributes[1], self.values[1][0]))
                if e1_flags[1][0] and e2_flags[1][0]:
                    # att2
                    judge_basis.append((self.attributes[2], self.values[2][0]))
                    if e1_flags[2][0] and e2_flags[2][0]:
                        return -1, judge_basis
                    elif (not e1_flags[2][0]) and e2_flags[2][0]:
                        return 0, judge_basis
                    elif (not e2_flags[2][0]) and e1_flags[2][0]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[2], self.values[2][1]))
                        if e1_flags[2][1] and e2_flags[2][1]:
                            return -1, judge_basis
                        elif (not e1_flags[2][1]) and e2_flags[2][1]:
                            return 0, judge_basis
                        elif (not e2_flags[2][1]) and e1_flags[2][1]:
                            return 1, judge_basis
                        else:
                            return -1, judge_basis
                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                    return 0, judge_basis
                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                    return 1, judge_basis
                else:
                    # att 1.1
                    judge_basis.append((self.attributes[1], self.values[1][1]))
                    if e1_flags[1][1] and e2_flags[1][1]:
                        # att2
                        judge_basis.append((self.attributes[2], self.values[2][0]))
                        if e1_flags[2][0] and e2_flags[2][0]:
                            return -1, judge_basis
                        elif (not e1_flags[2][0]) and e2_flags[2][0]:
                            return 0, judge_basis
                        elif (not e2_flags[2][0]) and e1_flags[2][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[2], self.values[2][1]))
                            if e1_flags[2][1] and e2_flags[2][1]:
                                return -1, judge_basis
                            elif (not e1_flags[2][1]) and e2_flags[2][1]:
                                return 0, judge_basis
                            elif (not e2_flags[2][1]) and e1_flags[2][1]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
                    elif (not e1_flags[1][1]) and e2_flags[1][1]:
                        return 0, judge_basis
                    elif (not e2_flags[1][1]) and e1_flags[1][1]:
                        return 1, judge_basis
                    else:
                        # att2
                        judge_basis.append((self.attributes[2], self.values[2][0]))
                        if e1_flags[2][0] and e2_flags[2][0]:
                            return -1, judge_basis
                        elif (not e1_flags[2][0]) and e2_flags[2][0]:
                            return 0, judge_basis
                        elif (not e2_flags[2][0]) and e1_flags[2][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[2], self.values[2][1]))
                            if e1_flags[2][1] and e2_flags[2][1]:
                                return -1, judge_basis
                            elif (not e1_flags[2][1]) and e2_flags[2][1]:
                                return 0, judge_basis
                            elif (not e2_flags[2][1]) and e1_flags[2][1]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
            elif (not e1_flags[0][0]) and e2_flags[0][0]:
                return 0, judge_basis
            elif (not e2_flags[0][0]) and e1_flags[0][0]:
                return 1, judge_basis
            else:
                # att0.1
                judge_basis.append((self.attributes[0], self.values[0][1]))
                if e1_flags[0][1] and e2_flags[0][1]:
                    # att 2.1
                    judge_basis.append((self.attributes[2], self.values[2][1]))
                    if e1_flags[2][1] and e2_flags[2][1]:
                        judge_basis.append((self.attributes[1], self.values[1][1]))
                        if e1_flags[1][1] and e2_flags[1][1]:
                            return -1, judge_basis
                        elif (not e1_flags[1][1]) and e2_flags[1][1]:
                            return 0, judge_basis
                        elif (not e2_flags[1][1]) and e1_flags[1][1]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[1], self.values[1][0]))
                            if e1_flags[1][0] and e2_flags[1][0]:
                                return -1, judge_basis
                            elif (not e1_flags[1][0]) and e2_flags[1][0]:
                                return 0, judge_basis
                            elif (not e2_flags[1][0]) and e1_flags[1][0]:
                                return 1, judge_basis
                            else:
                                return -1, judge_basis
                    elif (not e1_flags[2][1]) and e2_flags[2][1]:
                        return 0, judge_basis
                    elif (not e2_flags[2][1]) and e1_flags[2][1]:
                        return 1, judge_basis
                    else:
                        judge_basis.append((self.attributes[2], self.values[2][0]))
                        if e1_flags[2][0] and e2_flags[2][0]:
                            judge_basis.append((self.attributes[1], self.values[1][1]))
                            if e1_flags[1][1] and e2_flags[1][1]:
                                return -1, judge_basis
                            elif (not e1_flags[1][1]) and e2_flags[1][1]:
                                return 0, judge_basis
                            elif (not e2_flags[1][1]) and e1_flags[1][1]:
                                return 1, judge_basis
                            else:
                                judge_basis.append((self.attributes[1], self.values[1][0]))
                                if e1_flags[1][0] and e2_flags[1][0]:
                                    return -1, judge_basis
                                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                                    return 0, judge_basis
                                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                                    return 1, judge_basis
                                else:
                                    return -1, judge_basis
                        elif (not e1_flags[2][0]) and e2_flags[2][0]:
                            return 0, judge_basis
                        elif (not e2_flags[2][0]) and e1_flags[2][0]:
                            return 1, judge_basis
                        else:
                            judge_basis.append((self.attributes[1], self.values[1][1]))
                            if e1_flags[1][1] and e2_flags[1][1]:
                                return -1, judge_basis
                            elif (not e1_flags[1][1]) and e2_flags[1][1]:
                                return 0, judge_basis
                            elif (not e2_flags[1][1]) and e1_flags[1][1]:
                                return 1, judge_basis
                            else:
                                judge_basis.append((self.attributes[1], self.values[1][0]))
                                if e1_flags[1][0] and e2_flags[1][0]:
                                    return -1, judge_basis
                                elif (not e1_flags[1][0]) and e2_flags[1][0]:
                                    return 0, judge_basis
                                elif (not e2_flags[1][0]) and e1_flags[1][0]:
                                    return 1, judge_basis
                                else:
                                    return -1, judge_basis


class SingleValPreference():
    def __init__(self, pref_type, attributes, values, sampled_entities=[], questions=None, hard_questions=None,
                 keep_graph=False):
        # questions, hard_questions, question_max=100
        self.pref_type = pref_type
        self.attributes = attributes
        self.values = values
        self.formula = PrefFormula(pref_type=pref_type, attributes=attributes)
        self.num_atts = len(self.attributes)
        self.sampled_entities = sampled_entities
        self.questions = questions if questions else []
        self.hard_questions = hard_questions if hard_questions else []
        if keep_graph:
            self.pref_graph = self.gen_pref_graph()
        else:
            self.pref_graph = None

    def __str__(self):
        return "type: " + str(self.pref_type) + " atts: " + str(self.attributes) + " vals: " + str(self.values) + \
               "\nquestions:" + str(self.questions) + \
               "\ngraph:" + str(self.pref_graph)

    def update_questions(self, new_questions):
        self.questions = new_questions

    def update_hard_questions(self, new_hard_questions):
        self.hard_questions = new_hard_questions

    def gen_pref_graph(self):
        '''
        nodes: (num in graph, actual num)
        edge_type: 0: ordinate, 1: have, 2: att_preferred, 3: val_preferred
        edges: (head, type, tail)
        '''
        pref_graph = defaultdict(list)
        pref_graph["nodes"].append((0, "root"))
        if "UIUP" in self.pref_type:
            # "UIUP-1", "UIUP-2", "UIUP-3"
            for i in range(self.num_atts):
                # create nodes
                pref_graph["nodes"].append((i+1, self.attributes[i]))
                pref_graph["nodes"].append((2*i+1+self.num_atts, self.values[i]))
                pref_graph["nodes"].append((2*i+2+self.num_atts, self.values[i]))
                # create edges
                pref_graph["edges"].append((0, 0, i+1))
                pref_graph["edges"].append((i+1, 1, 2*i+1+self.num_atts))
                pref_graph["edges"].append((i+1, 1, 2*i+2+self.num_atts))
                pref_graph["edges"].append((2*i+1+self.num_atts, 3, 2*i+2+self.num_atts))
                if i >= 1:
                    pref_graph["edges"].append((i, 2, i+1))
        elif self.pref_type == "UICP-2":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[1]),
                (4, self.values[0]), (5, self.values[0]),
                (6, self.values[1]), (7, self.values[1]),
                (8, self.values[1]), (9, self.values[1]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (4, 0, 2), (5, 0, 3),
                (1, 1, 4), (1, 1, 5), (2, 1, 6), (2, 1, 7), (3, 1, 8), (3, 1, 9),
                (4, 3, 5), (6, 3, 7), (8, 3, 9)
            ]
        elif self.pref_type == "CIUP-3a":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.values[0]), (5, self.values[0]),
                (6, self.values[1]), (7, self.values[1]),
                (8, self.values[2]), (9, self.values[2]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (4, 0, 2), (5, 0, 3),
                (1, 1, 4), (1, 1, 5), (2, 1, 6), (2, 1, 7), (3, 1, 8), (3, 1, 9),
                (4, 3, 5), (6, 3, 7), (8, 3, 9),
            ]
        elif self.pref_type == "CIUP-3b":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.attributes[2]), (5, self.attributes[1]),
                (6, self.values[0]), (7, self.values[0]),
                (8, self.values[1]), (9, self.values[1]),
                (10, self.values[2]), (11, self.values[2]),
                (12, self.values[2]), (13, self.values[2]),
                (14, self.values[1]), (15, self.values[1]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (6, 0, 2), (6, 0, 3), (7, 0, 4), (7, 0, 5),
                (1, 1, 6), (1, 1, 7), (2, 1, 8), (2, 1, 9), (3, 1, 10), (3, 1, 11),
                (4, 1, 12), (4, 1, 13), (5, 1, 14), (5, 1, 15),
                (2, 2, 3), (4, 2, 5),
                (6, 3, 7), (8, 3, 9), (10, 3, 11), (12, 3, 13), (14, 3, 15),
            ]
        elif self.pref_type == "CICP-3":
            pref_graph["nodes"].extend([
                (1, self.attributes[0]), (2, self.attributes[1]), (3, self.attributes[2]),
                (4, self.attributes[2]), (5, self.attributes[1]),
                (6, self.values[0]), (7, self.values[0]),
                (8, self.values[1]), (9, self.values[1]),
                (10, self.values[2]), (11, self.values[2]),
                (12, self.values[2]), (13, self.values[2]),
                (14, self.values[1]), (15, self.values[1]),
            ])
            pref_graph["edges"] = [
                (0, 0, 1), (6, 0, 2), (6, 0, 3), (7, 0, 4), (7, 0, 5),
                (1, 1, 6), (1, 1, 7), (2, 1, 8), (2, 1, 9), (3, 1, 10), (3, 1, 11),
                (4, 1, 12), (4, 1, 13), (5, 1, 14), (5, 1, 15),
                (2, 2, 3), (4, 2, 5),
                (6, 3, 7), (8, 3, 9), (10, 3, 11), (12, 3, 13), (14, 3, 15),
            ]
        return pref_graph

    def judge_pair_priority(self, e1_info, e2_info):
        e1_flags = e1_info
        e2_flags = e2_info
        judge_basis = []
        if "UIUP" in self.pref_type:
            for i in range(self.num_atts):
                judge_basis.append(self.attributes[i])
                if e1_flags[i] != e2_flags[i]:
                    return e1_flags[i] > e2_flags[i], judge_basis
            return -1, judge_basis
        elif self.pref_type == "UICP-2":
            judge_basis.append(self.attributes[0])
            if e1_flags[0] != e2_flags[0]:
                return e1_flags[0] > e2_flags[0], judge_basis
            elif e1_flags[0] == True:
                judge_basis.append(self.attributes[1])
                if e1_flags[1] != e2_flags[1]:
                    return e1_flags[1] > e2_flags[1], judge_basis
                else:
                    return -1, judge_basis
            else:
                judge_basis.append(self.attributes[1])
                if e1_flags[1] != e2_flags[1]:
                    return e1_flags[1] < e2_flags[1], judge_basis # false > true
                else:
                    return -1, judge_basis
        elif self.pref_type == "CIUP-3a":
            judge_basis.append(self.attributes[0])
            if e1_flags[0] != e2_flags[0]:
                return e1_flags[0] > e2_flags[0], judge_basis
            elif e1_flags[0] == True:
                judge_basis.append(self.attributes[1])
                if e1_flags[1] != e2_flags[1]:
                    return e1_flags[1] > e2_flags[1], judge_basis
                else:
                    return -1, judge_basis
            else:
                judge_basis.append(self.attributes[2])
                if e1_flags[2] != e2_flags[2]:
                    return e1_flags[2] > e2_flags[2], judge_basis
                else:
                    return -1, judge_basis
        elif self.pref_type == "CIUP-3b":
            judge_basis.append(self.attributes[0])
            if e1_flags[0] != e2_flags[0]:
                return e1_flags[0] > e2_flags[0], judge_basis
            elif e1_flags[0] == True:
                judge_basis.append(self.attributes[1])
                if e1_flags[1] != e2_flags[1]:
                    return e1_flags[1] > e2_flags[1], judge_basis
                else:
                    judge_basis.append(self.attributes[2])
                    if e1_flags[2] != e2_flags[2]:
                        return e1_flags[2] > e2_flags[2], judge_basis
                    else:
                        return -1, judge_basis
            else:
                judge_basis.append(self.attributes[2])
                if e1_flags[2] != e2_flags[2]:
                    return e1_flags[2] > e2_flags[2], judge_basis
                else:
                    judge_basis.append(self.attributes[1])
                    if e1_flags[1] != e2_flags[1]:
                        return e1_flags[1] > e2_flags[1], judge_basis
                    else:
                        return -1, judge_basis
        else:
            # CICP-3
            judge_basis.append(self.attributes[0])
            if e1_flags[0] != e2_flags[0]:
                return e1_flags[0] > e2_flags[0], judge_basis
            elif e1_flags[0] == True:
                judge_basis.append(self.attributes[1])
                if e1_flags[1] != e2_flags[1]:
                    return e1_flags[1] > e2_flags[1], judge_basis
                else:
                    judge_basis.append(self.attributes[2])
                    if e1_flags[2] != e2_flags[2]:
                        return e1_flags[2] > e2_flags[2], judge_basis
                    else:
                        return -1, judge_basis
            else:
                judge_basis.append(self.attributes[2])
                if e1_flags[2] != e2_flags[2]:
                    return e1_flags[2] < e2_flags[2], judge_basis
                else:
                    judge_basis.append(self.attributes[1])
                    if e1_flags[1] != e2_flags[1]:
                        return e1_flags[1] < e2_flags[1], judge_basis
                    else:
                        return -1, judge_basis

    def serialize(self):
        return (self.pref_type, self.attributes, self.values, self.sampled_entities)

    @staticmethod
    def deserialize(info, keep_graph=False):
        return SingleValPreference(pref_type=info[0], attributes=info[1], values=info[2],
                                   sampled_entities=info[3], keep_graph=keep_graph)


class PrefFormula():
    '''
    The formula of preferences, including the preference type and the considered attributes
    '''
    def __init__(self, pref_type, attributes):
        self.pref_type = pref_type
        if "1" in pref_type:
            self.attributes = (attributes[0], )
        elif "2" in pref_type:
            self.attributes = (attributes[0], attributes[1])
        elif "3" in pref_type:
            self.attributes = (attributes[0], attributes[1], attributes[2])
        else:
            self.attributes = None
        self.entity_type = attributes[0][0]
        self.edge_index = self.gen_edge_idx()
        self.reverse_edge_idx = self.gen_reverse_edge_idx()
        self.rel_pos, self.vec_e_pos, self.vec_p_pos, self.vec_n_pos = self.gen_formula_info()
        self.p_pos = self.vec_p_pos[0]
        self.n_pos = self.vec_n_pos[0]
        for i in range(1, len(self.vec_p_pos)):
            self.p_pos += self.vec_p_pos[i]
            self.n_pos += self.vec_n_pos[i]
        self.node_num = len(self.rel_pos[0])

    def gen_formula_info(self):
        if self.pref_type == "UIUP-1":
            # (0, root), (1, rel1), (2, val1-Pos), (3, val1-Neg)
            rel_pos = [torch.LongTensor([0, 1, 1, 1])]
            vec_e_pos = torch.LongTensor([1, 1, 0, 0])
            vec_p_pos = [torch.LongTensor([0, 0, 1, 0])]
            vec_n_pos = [torch.LongTensor([0, 0, 0, 1])]

        elif self.pref_type == "UIUP-1-reverse":
            rel_pos = [torch.LongTensor([0, 1, 1, 1])]
            vec_e_pos = torch.LongTensor([1, 1, 0, 0])
            vec_p_pos = [torch.LongTensor([0, 0, 0, 1])]
            vec_n_pos = [torch.LongTensor([0, 0, 1, 0])]

        elif self.pref_type == "UIUP-2":
            # (0, root), (1, rel1), (2, rel2), (3, val1-Pos), (4, val1-Neg), (5, val2-Pos), (6, val2-Neg)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 1, 1, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 1, 1]),
            ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0])
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1])
            ]
        elif self.pref_type == "UIUP-2-reverse":
            rel_pos = [
                torch.LongTensor([0, 1, 0, 1, 1, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 1, 1]),
            ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0])
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1])
            ]

        elif self.pref_type == "UIUP-3":
            # (0, root), (1, rel1), (2, rel2), (3, rel3), (4, val1-Pos), (5, val1-Neg),
            # (6, val2-Pos), (7, val2-Neg), (8, val3-Pos), (9, val3-Neg)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0, 0, 1, 1]),
            ]
            # vec_e_pos = [
            #     torch.LongTensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
            # ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            ]
        elif self.pref_type == "UIUP-3-reverse":
            # (0, root), (1, rel1), (2, rel2), (3, rel3), (4, val1-Pos), (5, val1-Neg),
            # (6, val2-Pos), (7, val2-Neg), (8, val3-Pos), (9, val3-Neg)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0, 0, 1, 1]),
            ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            ]

        elif self.pref_type == "UICP-2":
            # (0, root), (1, rel1), (2, rel2), (3, rel2), (4, val1-Pos), (5, val1-Neg),
            # (6, val2-Pos), (7, val2-Neg), (8, val2-Neg), (9, val2-Pos)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]),
            ]
            # vec_e_pos = [
            #     torch.LongTensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            # ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 1]),
            ]
        elif self.pref_type == "CIUP-3a":
            # (0, root), (1, rel1), (2, rel2), (3, rel3), (4, val1-Pos), (5, val1-Neg),
            # (6, val2-Pos), (7, val2-Neg), (8, val3-Pos), (9, val3-Neg)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0, 0, 1, 1]),
            ]
            # vec_e_pos = [
            #     torch.LongTensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
            # ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            ]
        elif self.pref_type == "CIUP-3b":
            # (0, root), (1, rel1), (2, rel2), (3, rel3), (4, rel3), (5, rel2)
            # (6, val1-Pos), (7, val1-Neg),
            # (8, val2-Pos), (9, val2-Neg), (10, val3-Pos), (11, val3-Neg)
            # (12, val3-Pos), (13, val3-Neg), (14, val2-Pos), (15, val2-Neg)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]),
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            ]
            # vec_e_pos = [
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]),
            # ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]),
            ]
        elif self.pref_type == "CICP-3":
            # (0, root), (1, rel1), (2, rel2), (3, rel3), (4, rel3), (5, rel2)
            # (6, val1-Pos), (7, val1-Neg),
            # (8, val2-Pos), (9, val2-Neg), (10, val3-Pos), (11, val3-Neg)
            # (12, val3-Neg), (13, val3-Pos), (14, val2-Neg), (15, val2-Pos)
            rel_pos = [
                torch.LongTensor([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]),
                torch.LongTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            ]
            # vec_e_pos = [
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
            #     torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]),
            # ]
            vec_e_pos = torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            vec_p_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
            ]
            vec_n_pos = [
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
                torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
            ]
        return rel_pos, vec_e_pos, vec_p_pos, vec_n_pos

    def cudify(self):
        for i in range(len(self.rel_pos)):
            self.rel_pos[i] = self.rel_pos[i].cuda()
        for i in range(len(self.vec_p_pos)):
            self.vec_p_pos[i] = self.vec_p_pos[i].cuda()
        for i in range(len(self.vec_n_pos)):
            self.vec_n_pos[i] = self.vec_n_pos[i].cuda()

        self.vec_e_pos = self.vec_e_pos.cuda()

        self.p_pos = self.p_pos.cuda()
        self.n_pos = self.n_pos.cuda()


    def gen_edge_idx(self):
        '''
        nodes: (num in graph, actual num)
        edge_type: 1: ordinate, 2: have, 3: att_preferred, 4: val_preferred
        edges: (head, type, tail)
        '''

        if self.pref_type == "UIUP-1" or self.pref_type == "UIUP-1-reverse":
            edge_idx = [
                [5, 1, 0, 0],
                [0, 5, 2, 2],
                [0, 0, 5, 4],
                [0, 0, 0, 5],
            ]
        elif self.pref_type == "UIUP-2" or self.pref_type == "UIUP-2-reverse":
            edge_idx = [
                [5, 1, 1, 0, 0, 0, 0],
                [0, 5, 3, 2, 2, 0, 0],
                [0, 0, 5, 0, 0, 2, 2],
                [0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 5],
            ]
        elif self.pref_type == "UIUP-3" or self.pref_type == "UIUP-3-reverse":
            edge_idx = [
                [5, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 5, 3, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 5, 3, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 5, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            ]
        elif self.pref_type == "UICP-2":
            edge_idx = [
                [5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 5, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 2, 2],
                [0, 0, 1, 0, 5, 4, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            ]
        elif self.pref_type == "CIUP-3a":
            edge_idx = [
                [5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 5, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 2, 2],
                [0, 0, 1, 0, 5, 4, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            ]
        elif self.pref_type == "CIUP-3b":
            edge_idx = [
                [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 5, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                [0, 0, 1, 1, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            ]
        elif self.pref_type == "CICP-3":
            edge_idx = [
                [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 5, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                [0, 0, 1, 1, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            ]
        return edge_idx

    def gen_reverse_edge_idx(self):
        edge_idx = self.gen_edge_idx()
        for i in range(len(edge_idx)):
            for j in range(len(edge_idx[i])):
                if j > i:
                    t = edge_idx[j][i]
                    edge_idx[j][i] = edge_idx[i][j]
                    edge_idx[i][j] = t
        return edge_idx

    def __hash__(self):
        return hash((self.pref_type, self.attributes))

    def __eq__(self, other):
        return ((self.pref_type, self.attributes)) == ((other.pref_type, other.attributes))

    def __neq__(self, other):
        return ((self.pref_type, self.attributes)) != ((other.pref_type, other.attributes))

    def __str__(self):
        return self.pref_type + ": " + str(self.attributes)

class Formula():

    def __init__(self, query_type, rels):
        self.query_type = query_type
        self.target_mode = rels[0][0]
        self.rels = rels
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.anchor_modes = (rels[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.anchor_modes = tuple([rel[-1] for rel in rels])
        elif query_type == "3-inter_chain":
            self.anchor_modes = (rels[0][-1], rels[1][-1][-1])
        elif query_type == "3-chain_inter":
            self.anchor_modes = (rels[1][0][-1], rels[1][1][-1])

    def __hash__(self):
        return hash((self.query_type, self.rels))

    def __eq__(self, other):
        return ((self.query_type, self.rels)) == ((other.query_type, other.rels))

    def __neq__(self, other):
        return ((self.query_type, self.rels)) != ((other.query_type, other.rels))

    def __str__(self):
        return self.query_type + ": " + str(self.rels)


class Query():

    def __init__(self, query_graph, neg_samples, hard_neg_samples, neg_sample_max=100, keep_graph=False):
        query_type = query_graph[0]
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = (query_graph[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = tuple([query_graph[i][-1] for i in range(1, len(query_graph))])
        elif query_type == "3-inter_chain":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[1][-1], query_graph[2][-1][-1])
        elif query_type == "3-chain_inter":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[2][0][-1], query_graph[2][1][-1])
        self.target_node = query_graph[1][0]
        if keep_graph:
            self.query_graph = query_graph
        else:
            self.query_graph = None
        if not neg_samples is None:
            self.neg_samples = list(neg_samples) if len(neg_samples) < neg_sample_max else random.sample(neg_samples,
                                                                                                         neg_sample_max)
        else:
            self.neg_samples = None
        if not hard_neg_samples is None:
            self.hard_neg_samples = list(hard_neg_samples) if len(
                hard_neg_samples) <= neg_sample_max else random.sample(hard_neg_samples, neg_sample_max)
        else:
            self.hard_neg_samples = None

    def contains_edge(self, edge):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges = self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return edge in edges or (edge[1], _reverse_relation(edge[1]), edge[0]) in edges

    def get_edges(self):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges = self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return set(edges).union(set([(e[-1], _reverse_relation(e[1]), e[0]) for e in edges]))

    def __hash__(self):
        return hash((self.formula, self.target_node, self.anchor_nodes))

    def __eq__(self, other):
        return (self.formula, self.target_node, self.anchor_nodes) == (
        other.formula, other.target_node, other.anchor_nodes)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        if self.query_graph is None:
            raise Exception("Cannot serialize query loaded with query graph!")
        return (self.query_graph, self.neg_samples, self.hard_neg_samples)

    @staticmethod
    def deserialize(serial_info, keep_graph=False):
        return Query(serial_info[0], serial_info[1], serial_info[2],
                     None if serial_info[1] is None else len(serial_info[1]), keep_graph=keep_graph)


class Graph():
    """
    Simple container for heteregeneous graph data.
    """

    def __init__(self, features, feature_dims, relations, adj_lists):
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.full_sets = defaultdict(set)
        self.full_lists = {}
        self.meta_neighs = defaultdict(dict)
        for rel, adjs in self.adj_lists.items():
            full_set = set(self.adj_lists[rel].keys())
            self.full_sets[rel[0]] = self.full_sets[rel[0]].union(full_set)
        for mode, full_set in self.full_sets.items():
            self.full_lists[mode] = list(full_set)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def _make_flat_adj_lists(self):
        self.flat_adj_lists = defaultdict(lambda: defaultdict(list))
        for rel, adjs in self.adj_lists.items():
            for node, neighs in adjs.items():
                self.flat_adj_lists[rel[0]][node].extend([(rel, neigh) for neigh in neighs])

    def _cache_edge_counts(self):
        self.edges = 0.
        self.rel_edges = {}
        for r1 in self.relations:
            for r2 in self.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in list(self.adj_lists[rel].values()):
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += 1.
        self.rel_weights = OrderedDict()
        self.mode_edges = defaultdict(float)
        self.mode_weights = OrderedDict()
        for rel, edge_count in self.rel_edges.items():
            self.rel_weights[rel] = edge_count / self.edges
            self.mode_edges[rel[0]] += edge_count
        for mode, edge_count in self.mode_edges.items():
            self.mode_weights[mode] = edge_count / self.edges

    def remove_edges(self, edge_list):
        for edge in edge_list:
            try:
                self.adj_lists[edge[1]][edge[0]].remove(edge[-1])
            except Exception:
                continue

            try:
                self.adj_lists[_reverse_relation(edge[1])][edge[-1]].remove(edge[0])
            except Exception:
                continue
        self.meta_neighs = defaultdict(dict)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def get_all_edges(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form (node1, relation, node2)
        """
        edges = []
        random.seed(seed)
        for rel, adjs in self.adj_lists.items():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.items():
                edges.extend([(node, rel, neigh) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0,
                            exclude_rels=set([])):
        random.seed(seed)
        edges = defaultdict(list)
        for rel, adjs in self.adj_lists.items():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.items():
                edges[(rel,)].extend([(node, neigh) for neigh in neighs if neigh != -1])

    def get_negative_edge_samples(self, edge, num, rejection_sample=True):
        if rejection_sample:
            neg_nodes = set([])
            counter = 0
            while len(neg_nodes) < num:
                neg_node = random.choice(self.full_lists[edge[1][0]])
                if not neg_node in self.adj_lists[_reverse_relation(edge[1])][edge[2]]:
                    neg_nodes.add(neg_node)
                counter += 1
                if counter > 100 * num:
                    return self.get_negative_edge_samples(edge, num, rejection_sample=False)
        else:
            neg_nodes = self.full_sets[edge[1][0]] - self.adj_lists[_reverse_relation(edge[1])][edge[2]]
        neg_nodes = list(neg_nodes) if len(neg_nodes) <= num else random.sample(list(neg_nodes), num)
        return neg_nodes

    def sample_test_queries(self, train_graph, q_types, samples_per_type, neg_sample_max, verbose=True):
        queries = []
        for q_type in q_types:
            sampled = 0
            while sampled < samples_per_type:
                q = self.sample_query_subgraph_bytype(q_type)
                if q is None or not train_graph._is_negative(q, q[1][0], False):
                    continue
                negs, hard_negs = self.get_negative_samples(q)
                if negs is None or ("inter" in q[0] and hard_negs is None):
                    continue
                query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
                queries.append(query)
                sampled += 1
                if sampled % 1000 == 0 and verbose:
                    print("Sampled", sampled)
        return queries

    def sample_queries(self, arity, num_samples, neg_sample_max, verbose=True):
        sampled = 0
        queries = []
        while sampled < num_samples:
            q = self.sample_query_subgraph(arity)
            if q is None:
                continue
            negs, hard_negs = self.get_negative_samples(q)
            if negs is None or ("inter" in q[0] and hard_negs is None):
                continue
            query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
            queries.append(query)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print("Sampled", sampled)
        return queries

    def get_negative_samples(self, query):
        if query[0] == "3-chain" or query[0] == "2-chain":
            edges = query[1:]
            rels = [_reverse_relation(edge[1]) for edge in edges[::-1]]
            meta_neighs = self.get_metapath_neighs(query[-1][-1], tuple(rels))
            negative_samples = self.full_sets[query[1][1][0]] - meta_neighs
            if len(negative_samples) == 0:
                return None, None
            else:
                return negative_samples, None
        elif query[0] == "2-inter" or query[0] == "3-inter":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            for i in range(2, len(query)):
                rel = _reverse_relation(query[i][1])
                union_neighs = union_neighs.union(self.adj_lists[rel][query[i][-1]])
                inter_neighs = inter_neighs.intersection(self.adj_lists[rel][query[i][-1]])
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-inter_chain":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            chain_rels = [_reverse_relation(edge[1]) for edge in query[2][::-1]]
            chain_neighs = self.get_metapath_neighs(query[2][-1][-1], tuple(chain_rels))
            union_neighs = union_neighs.union(chain_neighs)
            inter_neighs = inter_neighs.intersection(chain_neighs)
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-chain_inter":
            inter_rel_1 = _reverse_relation(query[-1][0][1])
            inter_neighs_1 = self.adj_lists[inter_rel_1][query[-1][0][-1]]
            inter_rel_2 = _reverse_relation(query[-1][1][1])
            inter_neighs_2 = self.adj_lists[inter_rel_2][query[-1][1][-1]]

            inter_neighs = inter_neighs_1.intersection(inter_neighs_2)
            union_neighs = inter_neighs_1.union(inter_neighs_2)
            rel = _reverse_relation(query[1][1])
            pos_nodes = set([n for neigh in inter_neighs for n in self.adj_lists[rel][neigh]])
            union_pos_nodes = set([n for neigh in union_neighs for n in self.adj_lists[rel][neigh]])
            neg_samples = self.full_sets[query[1][1][0]] - pos_nodes
            hard_neg_samples = union_pos_nodes - pos_nodes
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples

    def sample_edge(self, node, mode):
        rel, neigh = random.choice(self.flat_adj_lists[mode][node])
        edge = (node, rel, neigh)
        return edge

    def sample_query_subgraph_bytype(self, q_type, start_node=None):
        if start_node is None:
            start_rel = random.choice(list(self.adj_lists.keys()))
            node = random.choice(list(self.adj_lists[start_rel].keys()))
            mode = start_rel[0]
        else:
            node, mode = start_node

        if q_type[0] == "3":
            if q_type == "3-chain" or q_type == "3-chain_inter":
                num_edges = 1
            elif q_type == "3-inter_chain":
                num_edges = 2
            elif q_type == "3-inter":
                num_edges = 3
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph_bytype(
                    "2-chain" if q_type == "3-chain" else "2-inter", start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if q_type[0] == "2":
            num_edges = 1 if q_type == "2-chain" else 2
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)

    def sample_query_subgraph(self, arity, start_node=None):
        if start_node is None:
            start_rel = random.choice(list(self.adj_lists.keys()))
            node = random.choice(list(self.adj_lists[start_rel].keys()))
            mode = start_rel[0]
        else:
            node, mode = start_node
        if arity > 3 or arity < 2:
            raise Exception("Only arity of at most 3 is supported for queries")

        if arity == 3:
            # 1/2 prob of 1 edge, 1/4 prob of 2, 1/4 prob of 3
            num_edges = random.choice([1, 1, 2, 3])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph(2, start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if arity == 2:
            num_edges = random.choice([1, 2])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)

    def get_metapath_neighs(self, node, rels):
        if node in self.meta_neighs[rels]:
            return self.meta_neighs[rels][node]
        current_set = [node]
        for rel in rels:
            current_set = set([neigh for n in current_set for neigh in self.adj_lists[rel][n]])
        self.meta_neighs[rels][node] = current_set
        return current_set

    def sample_preferences(self, entity_type, pref_type, num_pref_atts, num_samples, question_sample_max, verbose=True):
        '''
        todo: how to represent graph with multi-instances of one entity?
        preference: (("1-2"))
        '''

        sampled = 0
        preferences = []
        while sampled < num_samples:
            sampled_atts = [(entity_type, att[1], att[0]) for att in random.sample(self.relations[entity_type], num_pref_atts)]
            sampled_vals = []
            for att in sampled_atts:
                possible_entities = set([entity for entity, vals in self.adj_lists[att]])
                possible_values = set()
                for _, vals in self.adj_lists[att]:
                    possible_values.union(vals)
                sampled_vals.append(random.sample(possible_values, 2))

        return

    def sample_singlev_preferences(self, entity_type, pref_type, num_pref_atts, num_samples, question_sample_max,
                                   test_flag=False, train_graph=None, verbose=True):
        print("Sampling", pref_type)
        sampled = 0
        sampled_preferences = []
        # for _ in tqdm(range(num_samples)):
        while sampled < num_samples:
            starting_type = random.choice(entity_type)
            starting_node = random.choice(self.full_lists[starting_type])
            candidate_atts = [] #
            for rel, adjs in self.adj_lists.items():
                if starting_type == rel[0] and starting_node in adjs and adjs[starting_node]:
                    candidate_atts.append(rel)
            if len(candidate_atts) < num_pref_atts:
                continue
            sampled_atts = random.sample(candidate_atts, num_pref_atts)
            sampled_vals = [random.choice(list(self.adj_lists[att][starting_node])) for att in sampled_atts]

            '''
                        抽样 对每个可能值 抽10个候选元素
                        UIUP-1 2:   T > F
                        UIUP-1-reverse 2:   F > T
                        UIUP-2 4:   TT > TF > FT > FF
                        UIUP-2-reverse 4:   FT > FF > TT > TF
                        UIUP-3 8:   TTT > TTF > TFT > TFF > FTT > FTF > FFT > FFF
                        UIUP-3-reverse 8:   FTT > FTF > FFT > FFF > TTT > TTF > TFT > TFF
                        UICP-2 4:   TT > TF > FF > FT
                        CIUP-3a 4:  TTT = TTF > TFT = TFF > FTT = FFT > FTF = FFF
                        CIUP-3b 8:  TTT > TTF > TFT > TFF > FTT > FFT > FTF > FFF
                        CICP-3 8:   TTT > TTF > TFT > TFF > FFF > FTF > FFT > FTT
                        '''
            pos_entities = [set(self.adj_lists[_reverse_relation(sampled_atts[i])][sampled_vals[i]]) for i in
                            range(num_pref_atts)]
            sampled_entities = []
            p = set()
            for i in pos_entities:
                p.update(i)
            neg_entities = set(self.full_lists[starting_type]) - p
            if num_pref_atts == 3:
                ttt = list(pos_entities[0].intersection(pos_entities[1]).intersection(pos_entities[2]))
                ttf = list(pos_entities[0].intersection(pos_entities[1]) - pos_entities[2])
                tft = list(pos_entities[0].intersection(pos_entities[2]) - pos_entities[1])
                tff = list(pos_entities[0] - pos_entities[1] - pos_entities[2])
                ftt = list(pos_entities[1].intersection(pos_entities[2]) - pos_entities[0])
                ftf = list(pos_entities[1] - pos_entities[0] - pos_entities[2])
                fft = list(pos_entities[2] - pos_entities[0] - pos_entities[1])
                fff = list(neg_entities)

                if min(len(ttt), len(ttf), len(tft), len(tff), len(ftt), len(ftf), len(fft), len(fff)) == 0:
                    continue

                ttt = random.sample(ttt, min(len(ttt), 10))
                ttf = random.sample(ttf, min(len(ttf), 10))
                tft = random.sample(tft, min(len(tft), 10))
                tff = random.sample(tff, min(len(tff), 10))
                ftt = random.sample(ftt, min(len(ftt), 10))
                ftf = random.sample(ftf, min(len(ftf), 10))
                fft = random.sample(fft, min(len(fft), 10))
                fff = random.sample(fff, min(len(fff), 10))

                if pref_type == "CIUP-3a":
                    sampled_entities = [fff + ftf, ftt + fft, tff + tft, ttf + ttt]
                elif pref_type == "CIUP-3b":
                    sampled_entities = [fff, ftf, fft, ftt, tff, tft, ttf, ttt]
                elif pref_type == "CICP-3":
                    sampled_entities = [ftt, fft, ftf, fff, tff, tft, ttf, ttt]
                elif pref_type == "UIUP-3":
                    sampled_entities = [fff, fft, ftf, ftt, tff, tft, ttf, ttt]
                elif pref_type == "UIUP-3-reverse":
                    sampled_entities = [tff, tft, ttf, ttt, fff, fft, ftf, ftt]

            elif num_pref_atts == 2:
                tt = list(pos_entities[0].intersection(pos_entities[1]))
                tf = list(pos_entities[0] - pos_entities[1])
                ft = list(pos_entities[1] - pos_entities[0])
                ff = list(neg_entities)

                if min(len(tt), len(tf), len(ft), len(ff)) == 0:
                    continue

                tt = random.sample(tt, min(len(tt), 10))
                tf = random.sample(tf, min(len(tf), 10))
                ft = random.sample(ft, min(len(ft), 10))
                ff = random.sample(ff, min(len(ff), 10))
                if pref_type == "UICP-2":
                    sampled_entities = [ft, ff, tf, tt]
                elif pref_type == "UIUP-2":
                    sampled_entities = [ff, ft, tf, tt]
                elif pref_type == "UIUP-2-reverse":
                    sampled_entities = [tf, tt, ff, ft]

            else:
                # len = 1
                t = list(pos_entities[0])
                f = list(neg_entities)
                t = random.sample(t, min(len(t), 10))
                f = random.sample(f, min(len(f), 10))
                if pref_type == "UIUP-1":
                    sampled_entities = [t, f]
                elif pref_type == "UIUP-1-reverse":
                    sampled_entities = [f, t]

            for se in sampled_entities:
                if len(se) == 0:
                    continue
            preference = SingleValPreference(pref_type, sampled_atts, sampled_vals, sampled_entities=sampled_entities)
            sampled_preferences.append(preference)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print("Sampled:", sampled)


            # print("preference:", preference)
            # sample questions
            # sampled_questions = []
            # candidate_entities = [self.adj_lists[_reverse_relation(sampled_atts[i])][sampled_vals[i]] for i in range(num_pref_atts)]
            # candidate_for_question_2 = set()
            # candidate_for_question_1 = set()
            # candidate_for_question = set()
            # if len(candidate_entities) >= 3:
            #     candidate_for_question_2.update(candidate_entities[0].intersection(candidate_entities[1]))
            #     candidate_for_question_2.update(candidate_entities[1].intersection(candidate_entities[2]))
            #     candidate_for_question_2.update(candidate_entities[2].intersection(candidate_entities[0]))
            #     candidate_for_question.update(candidate_entities[0], candidate_entities[1], candidate_entities[2])
            #     candidate_for_question_1 = candidate_for_question - candidate_for_question_2
            #     if len(candidate_for_question_2) < math.sqrt(question_sample_max):
            #         continue
            # elif len(candidate_entities) == 2:
            #     candidate_for_question_2 = candidate_entities[0].intersection(candidate_entities[1])
            #     candidate_for_question.update(candidate_entities[0], candidate_entities[1])
            #     candidate_for_question_1 = candidate_for_question - candidate_for_question_2
            #     if len(candidate_for_question_2) < math.sqrt(question_sample_max):
            #         continue
            # else:
            #     candidate_for_question = self.full_lists[starting_type]
            #     if len(candidate_for_question) < math.sqrt(question_sample_max):
            #         continue
            # while len(sampled_questions) < question_sample_max:
            #     question = random.sample(candidate_for_question, 2)
            #     if len(candidate_entities) == 1:
            #         if len(sampled_questions) < question_sample_max * 7 / 8:
            #             if question[0] not in candidate_entities[0] and question[1] not in candidate_entities[0]:
            #                 continue
            #     else:
            #         # cq2 on
            #         if question[0] in candidate_for_question_1 and question[1] in candidate_for_question_1:
            #             continue
            #     e1_info = [question[0] in self.adj_lists[_reverse_relation(sampled_atts[i])][sampled_vals[i]] for i in
            #                range(num_pref_atts)]
            #     e2_info = [question[1] in self.adj_lists[_reverse_relation(sampled_atts[i])][sampled_vals[i]] for i in
            #                range(num_pref_atts)]
            #     answer, basis = preference.judge_pair_priority(e1_info, e2_info)
            #     '''
            #     TODO: 是否需要严格按照其移除？
            #     if test_flag:
            #         # at least one missing edge as basis.
            #         missing_edge_count = 0
            #         for att in basis:
            #             idx = sampled_atts.index(att)
            #             if e1_info[idx] and not train_graph._verify_edge(question[0], sampled_atts[idx], sampled_vals[idx]):
            #                 missing_edge_count += 1
            #         if missing_edge_count == 0 or missing_edge_count > 2:
            #             continue
            #     '''
            #     # print("question:", question, "e1_info:", e1_info, "e2_info:", e2_info, "answer:", answer, "basis:", basis)
            #     sampled_questions.append((question, answer))
            # preference.update_questions(sampled_questions)
            # sampled_preferences.append(preference)
            # sampled += 1
            # if sampled % 1000 == 0 and verbose:
            #     print("Sampled:", sampled)
        print("in pref_graph sampled preferences:", len(sampled_preferences))
        return sampled_preferences


    ## TESTING CODE

    def _check_edge(self, query, i):
        return query[i][-1] in self.adj_lists[query[i][1]][query[i][0]]

    def _is_subgraph(self, query, verbose):
        if query[0] == "3-chain":
            for i in range(3):
                if not self._check_edge(query, i + 1):
                    raise Exception(str(query))
            if not (query[1][-1] == query[2][0] and query[2][-1] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "2-chain":
            for i in range(2):
                if not self._check_edge(query, i + 1):
                    raise Exception(str(query))
            if not query[1][-1] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "2-inter":
            for i in range(2):
                if not self._check_edge(query, i + 1):
                    raise Exception(str(query))
            if not query[1][0] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "3-inter":
            for i in range(3):
                if not self._check_edge(query, i + 1):
                    raise Exception(str(query))
            if not (query[1][0] == query[2][0] and query[2][0] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "3-inter_chain":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][0] == query[2][0][0] and query[2][0][-1] == query[2][1][0]):
                raise Exception(str(query))
        if query[0] == "3-chain_inter":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][-1] == query[2][0][0] and query[2][0][0] == query[2][1][0]):
                raise Exception(str(query))
        return True

    def _is_negative(self, query, neg_node, is_hard):
        if query[0] == "2-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            if query[2][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1])):
                return False
        if query[0] == "3-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2], query[3])
            if query[3][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1], query[3][1])):
                return False
        if query[0] == "2-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2)) or not (
                        self._check_edge(query, 1) or self._check_edge(query, 2)):
                    return False
        if query[0] == "3-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]),
                     (neg_node, query[3][1], query[3][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3)) \
                        or not (self._check_edge(query, 1) or self._check_edge(query, 2) or self._check_edge(query, 3)):
                    return False
        if query[0] == "3-inter_chain":
            query = (
            query[0], (neg_node, query[1][1], query[1][2]), ((neg_node, query[2][0][1], query[2][0][2]), query[2][1]))
            meta_check = lambda: query[2][-1][-1] in self.get_metapath_neighs(query[1][0],
                                                                              (query[2][0][1], query[2][1][1]))
            neigh_check = lambda: self._check_edge(query, 1)
            if not is_hard:
                if meta_check() and neigh_check():
                    return False
            else:
                if (meta_check() and neigh_check()) or not (meta_check() or neigh_check()):
                    return False
        if query[0] == "3-chain_inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            target_neigh = self.adj_lists[query[1][1]][neg_node]
            neigh_1 = self.adj_lists[_reverse_relation(query[2][0][1])][query[2][0][-1]]
            neigh_2 = self.adj_lists[_reverse_relation(query[2][1][1])][query[2][1][-1]]
            if not is_hard:
                if target_neigh in neigh_1.intersection(neigh_2):
                    return False
            else:
                if target_neigh in neigh_1.intersection(neigh_2) and not target_neigh in neigh_1.union(neigh_2):
                    return False
        return True

    def _run_test(self, num_samples=1000):
        for i in range(num_samples):
            q = self.sample_query_subgraph(2)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
            q = self.sample_query_subgraph(3)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
        return True

    def _verify_edge(self, head, rel, tail):
        if tail in self.adj_lists[rel][head]:
            return True
        else:
            return False

    """
    TO DELETE? 
    def sample_chain_from_node(self, length, node, rel):
        rels = [rel]
        for cur_len in range(length-1):
            next_rel = random.choice(self.relations[rels[-1][-1]])
            rels.append((rels[-1][-1], next_rel[-1], next_rel[0]))

        rels = tuple(rels)
        meta_neighs = self.get_metapath_neighs(node, rels)
        rev_rel = _reverse_relation(rels[-1])
        full_set = self.full_sets[rev_rel]
        diff_set = full_set - meta_neighs
        if len(meta_neighs) == 0 or len(diff_set) == 0:
            return None, None, None
        chain = (node, random.choice(list(meta_neighs)))
        neg_chain = (node, random.choice(list(diff_set)))
        return chain, neg_chain, rels

    def sample_chain(self, length, start_mode):
        rel = random.choice(self.relations[start_mode])
        rel = (start_mode, rel[-1], rel[0])
        if len(self.adj_lists[rel]) == 0:
            return None, None, None
        node = random.choice(self.adj_lists[rel].keys())
        return self.sample_chain_from_node(length, node, rel)

    def sample_chains(self, length, anchor_weights, num_samples):
        sampled = 0
        graph_chains = defaultdict(list)
        neg_chains = defaultdict(list)
        while sampled < num_samples: 
            anchor_mode = anchor_weights.keys()[np.argmax(np.random.multinomial(1, anchor_weights.values()))]
            chain, neg_chain, rels = self.sample_chain(length, anchor_mode)
            if chain is None:
                continue
            graph_chains[rels].append(chain)
            neg_chains[rels].append(neg_chain)
            sampled += 1
        return graph_chains, neg_chains


    def sample_polytree_rootinter(self, length, target_mode, try_out=100):
        num_chains = random.randint(2,length)
        added = 0
        nodes = []
        rels_list = []

        for i in range(num_chains):
            remaining = length-added-num_chains
            if i != num_chains - 1:
                remaining = remaining if remaining == 0 else random.randint(0, remaining)
            added += remaining
            chain_len = 1 + remaining
            if i == 0:
                chain, _, rels = self.sample_chain(chain_len, target_mode)
                try_count = 0 
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain(chain_len, target_mode)
                    try_count += 1

                if chain is None:
                    return None, None, None, None, None
                target_node = chain[0]
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))
            else:
                rel = random.choice([r for r in self.relations[target_mode] 
                    if len(self.adj_lists[(target_mode, r[-1], r[0])][target_node]) > 0])
                rel = (target_mode, rel[-1], rel[0])
                chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                try_count = 0
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                    if chain is None:
                        try_count += 1
                    elif chain[-1] in nodes:
                        chain = None
                if chain is None:
                    return None, None, None, None, None
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))

        for i in range(len(nodes)):
            meta_neighs = self.get_metapath_neighs(nodes[i], rels_list[i])
            if i == 0:
                meta_neighs_inter = meta_neighs
                meta_neighs_union = meta_neighs
            else:
                meta_neighs_inter = meta_neighs_inter.intersection(meta_neighs)
                meta_neighs_union = meta_neighs_union.union(meta_neighs)
        hard_neg_nodes = list(meta_neighs_union-meta_neighs_inter)
        neg_nodes = list(self.full_sets[rels[0]]-meta_neighs_inter)
        if len(neg_nodes) == 0:
            return None, None, None, None, None
        if len(hard_neg_nodes) == 0:
            return None, None, None, None, None

        return target_node, neg_nodes, hard_neg_nodes, tuple(nodes), tuple(rels_list)


    def sample_polytrees_parallel(self, length, thread_samples, threads, try_out=100):
        pool = Pool(threads)
        sample_func = partial(self.sample_polytree, length)
        sizes = [thread_samples for _ in range(threads)]
        results = pool.map(sample_func, sizes)
        polytrees = {}
        neg_polytrees = {}
        hard_neg_polytrees = {}
        for p, n, h in results: 
            polytrees.update(p)
            neg_polytrees.update(n)
            hard_neg_polytrees.updarte(h)
        return polytrees, neg_polytrees, hard_neg_polytrees

    def sample_polytrees(self, length, num_samples, try_out=1):
        samples = 0
        polytrees = defaultdict(list)
        neg_polytrees = defaultdict(list)
        hard_neg_polytrees = defaultdict(list)
        while samples < num_samples:
            t, n, h_n, nodes, rels = self.sample_polytree(length, random.choice(self.relations.keys()))
            if t is None:
                continue
            samples += 1
            polytrees[rels].append((t, nodes))
            neg_polytrees[rels].append((n, nodes))
            hard_neg_polytrees[rels].append((h_n, nodes))
        return polytrees, neg_polytrees, hard_neg_polytrees

    """

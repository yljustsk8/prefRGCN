import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, RGATConv
from torch_geometric.nn import global_add_pool, global_mean_pool

import time

class PrefRGCN(nn.Module):
    '''

    '''
    def __init__(self, gq_model, gq_dec_type, emb_dim, hidden, rel_num=4):

        super(PrefRGCN, self).__init__()
        self.gq_model = gq_model
        if gq_dec_type == "bilinear":
            self.rel_proj = nn.Linear(emb_dim, 1) # (rel_dim, rel_dim) ==> (rel_dim, 1)
        else:
            self.rel_proj = nn.Linear(emb_dim, emb_dim) # (rel_dim, 1) ==> (rel_dim, 1)
        self.positive_proj = nn.Linear(emb_dim, hidden) # (1, node_dim) ==> (node_dim, hidden)
        self.negative_proj = nn.Linear(emb_dim, hidden) # (1, node_dim) ==> (node_dim, hidden)
        self.other_proj = nn.Linear(emb_dim, hidden)  # (1, node_dim) ==> (node_dim, hidden)
        self.reproj = nn.Linear(hidden, emb_dim) # (1, hidden) ==> (1, node_dim)
        # 映射
        # self.rel_proj = torch.FloatTensor(rel_dim, 1) # (rel_dim, rel_dim) ==> (rel_dim, 1)
        # self.positive_proj = torch.FloatTensor(rel_dim + node_dim, hidden) # (1, rel_dim + node_dim) ==> (rel_dim + node_dim, hidden)
        # self.negative_proj = torch.FloatTensor(rel_dim + node_dim, hidden) # (1, rel_dim + node_dim) ==> (rel_dim + node_dim, hidden)
        # self.reproj = torch.FloatTensor(hidden, node_dim) # (1, hidden) ==> (1, node_dim)

        # GNN
        # self.conv1 = GCNConv(hidden, hidden)
        # self.conv2 = GCNConv(hidden, hidden)
        self.conv1 = RGCNConv(
            in_channels=hidden,
            out_channels=hidden,
            num_relations=5,
            num_bases=4
        )
        self.conv2 = RGCNConv(
            in_channels=hidden,
            out_channels=hidden,
            num_relations=5,
            num_bases=4
        )
        self.conv3 = RGCNConv(
            in_channels=hidden,
            out_channels=hidden,
            num_relations=5,
            num_bases=4
        )

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, formula, prefs, edge_index, edge_type, batch, mask, targets):
        '''

        targets: [bs]
        '''

        atts = formula.attributes
        vals = [] # [vals_num, bs]
        node_embeds = [] # [vals_num, bs, dim]
        rel_embeds = [] # [rels_num, dim]

        targets_embeds = self.gq_model.enc.forward(targets, atts[0][0]).t() # [bs, dim]

        for i in range(len(atts)):
            vals.append([pref.values[i] for pref in prefs])

        node_num = formula.node_num
        bs = len(vals[0])
        # print("node num:", node_num, "bs:", bs)

        for i in range(len(atts)):
            node_embeds.append(self.gq_model.enc.forward(vals[i], atts[i][-1]).t().unsqueeze(0).repeat(node_num, 1, 1))
            rel_embeds.append(self.rel_proj(self.gq_model.path_dec.mats[atts[i]]).t().unsqueeze(0).repeat(formula.node_num, len(prefs), 1))

        emb = rel_embeds[0] * formula.rel_pos[0].unsqueeze(-1).unsqueeze(-1) + \
            node_embeds[0] * (formula.vec_p_pos[0] + formula.vec_n_pos[0]).unsqueeze(-1).unsqueeze(-1)
        # [node_num, bs, emb_dim]
        for i in range(1, len(atts)):
            emb += rel_embeds[i] * formula.rel_pos[i].unsqueeze(-1).unsqueeze(-1) + \
            node_embeds[i] * (formula.vec_p_pos[i] + formula.vec_n_pos[i]).unsqueeze(-1).unsqueeze(-1)

        # 肯定点+否定点+边
        x = self.positive_proj(emb) * formula.p_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.negative_proj(emb) * formula.n_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.other_proj(emb) * formula.vec_e_pos.unsqueeze(-1).unsqueeze(-1)
        # [node_num, bs, hidden]
        x = x.transpose(1, 0).reshape(node_num * bs, -1)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        # x = global_add_pool(x * mask.unsqueeze(-1), batch)
        x = global_add_pool(x, batch)

        x = self.reproj(x)

        score = (x * targets_embeds).sum(-1)
        return score

    def margin_loss(self, formula, prefs, edge_index, edge_type, batch, mask, margin=1):
        t = random.randint(0, len(prefs[0].sampled_entities)-2)
        pos_nodes = [random.choice(pref.sampled_entities[t+1]) for pref in prefs]
        neg_nodes = [random.choice(pref.sampled_entities[t]) for pref in prefs]
        pos_scores = self.forward(formula, prefs, edge_index, edge_type, batch, mask, targets=pos_nodes)
        neg_scores = self.forward(formula, prefs, edge_index, edge_type, batch, mask, targets=neg_nodes)
        loss = margin - (pos_scores - neg_scores)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss

    def contrast_loss(self, formula, prefs, edge_index, edge_type, batch, mask):
        atts = formula.attributes
        vals = []  # [vals_num, bs]
        node_embeds = []  # [vals_num, bs, dim]
        rel_embeds = []  # [rels_num, dim]

        for i in range(len(atts)):
            vals.append([pref.values[i] for pref in prefs])

        node_num = formula.node_num
        bs = len(vals[0])

        for i in range(len(atts)):
            node_embeds.append(self.gq_model.enc.forward(vals[i], atts[i][-1]).t().unsqueeze(0).repeat(node_num, 1, 1))
            rel_embeds.append(
                self.rel_proj(self.gq_model.path_dec.mats[atts[i]]).t().unsqueeze(0).repeat(formula.node_num,
                                                                                            len(prefs), 1))

        emb = rel_embeds[0] * formula.rel_pos[0].unsqueeze(-1).unsqueeze(-1) + \
              node_embeds[0] * (formula.vec_p_pos[0] + formula.vec_n_pos[0]).unsqueeze(-1).unsqueeze(-1)

        # [node_num, bs, emb_dim]
        for i in range(1, len(atts)):
            emb += rel_embeds[i] * formula.rel_pos[i].unsqueeze(-1).unsqueeze(-1) + \
                   node_embeds[i] * (formula.vec_p_pos[i] + formula.vec_n_pos[i]).unsqueeze(-1).unsqueeze(-1)

        # print("edge_index:", edge_index)
        # print("edge_type:", edge_type)
        # print("emb:", emb)

        # 肯定点+否定点+边
        x = self.positive_proj(emb) * formula.p_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.negative_proj(emb) * formula.n_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.other_proj(emb) * formula.vec_e_pos.unsqueeze(-1).unsqueeze(-1)

        x1 = self.positive_proj(emb) * formula.n_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.negative_proj(emb) * formula.p_pos.unsqueeze(-1).unsqueeze(-1) + \
            self.other_proj(emb) * formula.vec_e_pos.unsqueeze(-1).unsqueeze(-1)

        # [node_num, bs, hidden]
        x = x.transpose(1, 0).reshape(node_num * bs, -1)
        # print("x:", x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        # print("x':", x)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        # print("x'':", x)
        x = global_add_pool(x * mask.unsqueeze(-1), batch)
        # x = global_add_pool(x, batch)

        x1 = x1.transpose(1, 0).reshape(node_num * bs, -1)
        # print("x1:", x1)
        x1 = F.relu(self.conv1(x1, edge_index, edge_type))
        # print("x1':", x1)
        x1 = F.relu(self.conv2(x1, edge_index, edge_type))
        x1 = self.conv3(x1, edge_index, edge_type)
        # print("x1'':", x1)
        x1 = global_add_pool(x1 * mask.unsqueeze(-1), batch)
        # x1 = global_add_pool(x1, batch)

        loss = 1 + self.cos(x, x1)
        loss = loss.mean()
        return loss
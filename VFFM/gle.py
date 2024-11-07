from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GATConv, GATv2Conv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models.basic_gnn import *
import os

# Architecture abstractions
#
'''
GNN model has three parameters: GNN, pooling layer, cfg parameters
'''
class GNNClassifier(nn.Module):

    def __init__(self, gnn: nn.Module, pooling_layer: nn.Module, cfg) -> None:
        super().__init__()
        self.gnn = gnn
        self.polling = pooling_layer
        self.cfg = cfg

    def forward(self, batched_data):
        '''
            Params
            ---
            * `batched_data`: The batch as outputted from torch_geometric dataloader
        '''
        # We might pass more features that gnn_node features
        # Those would be aux features for the pooling layer


        batched_x = batched_data.x[:, :self.cfg.gnn_node_features]
        aux_feats = batched_data.x[:, self.cfg.gnn_node_features:]

        # standardize x. Sometimes m,v are 0 and 1 so no std happens
        batched_x = (batched_x - self.cfg.m)/self.cfg.v

        args = {}
        args['lesion_wide_feat_tensor'] = batched_data.lesion_wide_feat_tensor

        out = self.gnn(batched_x,batched_data.edge_index)
        out = self.polling(out, batched_data.batch, aux_feats, **args)

        return out

    def prep_predictions(self, pred, to_cpu:bool=True):
        ''' Prepare predictions for validation bases on model structure '''
        if self.cfg.nb_classes > 1:
            return torch.argmax(torch.softmax(pred.cpu().detach(), dim=1), dim=1)
        else:
            return torch.sigmoid(pred.cpu().detach())

#
# GNN's
#
def gnn_resolver(cfg) -> BasicGNN:
    import sys
    gnn_class_name = getattr(sys.modules[__name__], cfg.gnn_cls_name)
    
    # TODO add also our models
    # Dont add output channels since we will pool after
    return gnn_class_name(
        in_channels = cfg.gnn_node_features,
        hidden_channels = cfg.gnn_node_emb_dim,
        num_layers = cfg.gnn_nb_layers,
        dropout = cfg.gnn_dropout,
        act = cfg.gnn_act,
        norm = cfg.gnn_norm
    )


# Multi purpose pooling Layers
def all_statistics_pooling(features, batch):
    assert features.shape[0] == len(batch)
    m = global_mean_pool(features, batch)
    ma = global_max_pool(features, batch)
    mi = -global_max_pool(-features, batch)
    std = global_mean_pool(features ** 2, batch) - global_mean_pool(features, batch) ** 2
    return torch.hstack((m, mi, ma, std))

def resolve_node_pooling(node_pooling):
    if node_pooling == "all_stats":
        return all_statistics_pooling, 4
    elif node_pooling == "sum":
        return global_add_pool, 1
    elif node_pooling == "max":
        return global_max_pool, 1
    elif node_pooling == "mean":
        return global_mean_pool, 1
    else:
        raise TypeError(f"Unknown node pooling type: {node_pooling}")

class ConfigurablePooling(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_lesion_mask = cfg.pool_only_lesion_points
        self.node_pooling, pooling_mul = resolve_node_pooling(cfg.node_pooling)
        self.gnn_use_global_info = cfg.use_lesion_wide_info
        self.is_siamese = cfg.gnn_is_siamese
        self.gnn_node_emb_dim = cfg.gnn_node_emb_dim
        in_dim = cfg.gnn_node_emb_dim
        out_dim = cfg.nb_classes
        global_info_dim = cfg.lesion_wide_feat_dim if self.gnn_use_global_info else 0

        # self.siamese_reshaping = in_dim * pooling_mul

        self.lin1= nn.Sequential(
            nn.Linear(in_dim * pooling_mul , 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
        )

        # lin2 might also get global info concatenated
        self.lin2= nn.Sequential(
            nn.Linear(64 + global_info_dim, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(16, 1),
        )



    def forward(self, node_embs, batch_indexes, lesion_mask = None, lesion_wide_feat_tensor=None):
        # lesion masking if requested
        if self.use_lesion_mask:
           
           node_embs = node_embs[lesion_mask]
           batch_indexes = batch_indexes[lesion_mask]

        # node pooling
        out_graph = self.node_pooling(node_embs, batch_indexes) # batch, 256

        out = self.lin1(out_graph)
        # global info attachment
        out = torch.hstack([out, lesion_wide_feat_tensor]) # type: ignore
        view_pre = self.lin2(out)

        out = self.lin2[:2](out)

        # out = self.lin1[:5](out_graph)

        return out,view_pre


#
# Util Layers
#

class PoolingLayer(nn.Module):
    def __init__(self, d, global_hidden_dim):
        """ Pool node or edge features to graph level features. """
        super().__init__()
        self.lin = nn.Sequential(
                nn.Linear(4 * d, global_hidden_dim),
                nn.BatchNorm1d(global_hidden_dim),
                nn.ReLU(),
                nn.Linear(global_hidden_dim, global_hidden_dim)
        )

    def forward(self, features, batch):
        assert features.shape[0] == len(batch)
        m = global_mean_pool(features, batch)
        ma = global_max_pool(features, batch)
        mi = -global_max_pool(-features, batch)
        std = global_mean_pool(features ** 2, batch) - global_mean_pool(features, batch) ** 2
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out
    

def get_model(cfg):
    ''' Resolves if we use a CustomGIN or one from torch_geometric'''
    return GNNClassifier(gnn_resolver(cfg),
                    ConfigurablePooling(cfg), cfg)


class GNNClassificationModule(nn.Module):
    def __init__(self,
        cfg, 
        
    ):
        super().__init__()
        self.cfg = cfg

        # model construction
        self.model = get_model(cfg)

    def forward(self, data):

        return self.model(data)
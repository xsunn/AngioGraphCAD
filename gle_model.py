from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GATConv, GATv2Conv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models.basic_gnn import *
import random
#
# Architecture abstractions
#
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GATConv, GATv2Conv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models.basic_gnn import *
import argparse

#
# Architecture abstractions
#

class GNNClassifier(nn.Module):

    def __init__(self, gnn: nn.Module, pooling_layer: nn.Module, cfg) -> None:
        super().__init__()
        self.gnn = gnn
        self.pooling = pooling_layer
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
        if hasattr(batched_data, 'siamese_ptr'):
            args['siamese_ptr'] = batched_data.siamese_ptr
        if hasattr(batched_data, 'lesion_wide_feat_tensor'):
            args['lesion_wide_feat_tensor'] = batched_data.lesion_wide_feat_tensor

        out = self.gnn(batched_x, batched_data.edge_index)
        out = self.pooling(out, batched_data.batch, aux_feats, **args)
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


#
#  Pooling
#

class SiamesePoolingWithFFR(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        # Create one node_pooling and then a linear for classification
        layers = []
        factor = cfg.gnn_ffr_pooling_factor
        min_dim = cfg.gnn_ffr_pooling_proj_dim
        dim = 2 * cfg.gnn_global_hidden_dim
        if factor == 1:
            layers.append(nn.Linear(dim, min_dim))
            layers.extend([nn.BatchNorm1d(min_dim), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
            dim = min_dim
        else:
            while dim//factor >= min_dim:
                layers.append(nn.Linear(dim, dim//factor))
                layers.extend([nn.BatchNorm1d(dim//factor), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
                dim //= factor

        # Create one node_pooling and then a linear for classification
        self.node_pooling = PoolingLayer(cfg.gnn_node_emb_dim, cfg.gnn_global_hidden_dim)
        self.projection = nn.Sequential(*layers)
        self.final_layer = nn.Linear(dim+1, cfg.nb_classes)


    @staticmethod
    def _ffr_per_batch(aux_feats: torch.Tensor, batch_indexes: torch.Tensor, lesion_mask: torch.Tensor):
        prev_val = 1.2 # ffr goes until 1.0
        idx = 0
        ffr_measures = torch.zeros((batch_indexes.max() + 1,1)).to(aux_feats.device)
        for v in aux_feats[lesion_mask]:
            if v.item() != prev_val:
                ffr_measures[idx] = v
                idx +=1
                prev_val = v.item()
        return ffr_measures

    def forward(self, x:torch.Tensor, batch_indexes: torch.Tensor, aux_feats: Optional[torch.Tensor] = None, **kwargs):
        '''
            Params
            ---
            * `x`: Thea batched embedding representation from the gnn: Size: [ |Nodes in batch| x | Node Embedding Dim | ]
            * `batch_indexes`: A tensor with size: | Nodes in Batch | that has 0,1 values based on which nodes appears in what batch
            * `aux_feats`: A tensor with size: [|Nodes in Batch| x 1 ]. The value on each item is FFR or 0.0 if that point is not a lesion.
        '''
        # Re create batch_indexes
        assert 'siamese_ptr' in kwargs, "No siamese ptrs"
        siamese_ptr = kwargs['siamese_ptr']
        new_batch_indexes = torch.zeros(x.shape[0], dtype=torch.int64, device=batch_indexes.device)
        curr_idx = 0
        curr_dpoints = 0
        for ptrs in siamese_ptr:
            range_v1_st, range_v1_end = curr_dpoints, ptrs[1] + curr_dpoints
            range_v2_st, range_v2_end = range_v1_end + 1, ptrs[2] + curr_dpoints
            curr_dpoints =  range_v2_end + 1

            new_batch_indexes[range_v1_st:range_v1_end+1] = curr_idx
            curr_idx += 1
            new_batch_indexes[range_v2_st:range_v2_end+1] = curr_idx
            curr_idx += 1

        # get ffr per batch
        lesion_mask = (aux_feats != 0).view(-1)
        ffr_per_batch = self._ffr_per_batch(aux_feats, batch_indexes, lesion_mask)

        # Pool all global vectors
        x_out = self.node_pooling(x, new_batch_indexes)
        # concat siamese vectors (every 2 vectors are siamese)
        x_out = x_out.view(-1, 2 * self.cfg.gnn_global_hidden_dim)

        out = F.relu(x_out)
        out = self.projection(out)
        out = torch.hstack([out, ffr_per_batch])
        out = self.final_layer(out)
        return out

class SiamesePooling(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        # Create one node_pooling and then a linear for classification
        layers = []
        factor = cfg.gnn_ffr_pooling_factor
        min_dim = cfg.gnn_ffr_pooling_proj_dim
        dim = 2 * cfg.gnn_global_hidden_dim
        if factor == 1:
            layers.append(nn.Linear(dim, min_dim))
            layers.extend([nn.BatchNorm1d(min_dim), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
            dim = min_dim
        else:
            while dim//factor >= min_dim:
                layers.append(nn.Linear(dim, dim//factor))
                layers.extend([nn.BatchNorm1d(dim//factor), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
                dim //= factor
        layers.append(nn.Linear(dim, cfg.nb_classes))

        # Create one node_pooling and then a linear for classification
        self.node_pooling = PoolingLayer(cfg.gnn_node_emb_dim, cfg.gnn_global_hidden_dim)
        self.final_linear= nn.Sequential(*layers)

    def forward(self, x:torch.Tensor, batch_indexes: torch.Tensor, aux_feats: Optional[torch.Tensor] = None, **kwargs):
        '''
            Params
            ---
            * `x`: Thea batched embedding representation from the gnn: Size: [ |Nodes in batch| x | Node Embedding Dim | ]
            * `batch_indexes`: A tensor with size: | Nodes in Batch | that has 0,1 values based on which nodes appears in what batch
            * `aux_feats`: A tensor with size: [|Nodes in Batch| x 1 ]. The value on each item is FFR or 0.0 if that point is not a lesion.
        '''
        # Re create batch_indexes
        assert 'siamese_ptr' in kwargs, "No siamese ptrs"
        siamese_ptr = kwargs['siamese_ptr']
        new_batch_indexes = torch.zeros(x.shape[0], dtype=torch.int64, device=batch_indexes.device)
        curr_idx = 0
        curr_dpoints = 0
        for ptrs in siamese_ptr:
            range_v1_st, range_v1_end = curr_dpoints, ptrs[1] + curr_dpoints
            range_v2_st, range_v2_end = range_v1_end + 1, ptrs[2] + curr_dpoints
            curr_dpoints =  range_v2_end + 1

            new_batch_indexes[range_v1_st:range_v1_end+1] = curr_idx
            curr_idx += 1
            new_batch_indexes[range_v2_st:range_v2_end+1] = curr_idx
            curr_idx += 1

        # Pool all global vectors
        x_out = self.node_pooling(x, new_batch_indexes)
        # concat siamese vectors (every 2 vectors are siamese)
        x_out = x_out.view(-1, 2 * self.cfg.gnn_global_hidden_dim)

        out = F.relu(x_out)
        out = self.final_linear(out)
        return out


class LesionPoolingWithClinicalData(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        # Crate decreasing list of models
        layers = []
        factor = cfg.gnn_ffr_pooling_factor
        min_dim = cfg.gnn_ffr_pooling_proj_dim
        dim = cfg.gnn_global_hidden_dim
        if factor == 1:
            layers.append(nn.Linear(dim, min_dim))
            layers.extend([nn.BatchNorm1d(min_dim), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
            dim = min_dim
        else:
            while dim//factor >= min_dim:
                layers.append(nn.Linear(dim, dim//factor))
                layers.extend([nn.BatchNorm1d(dim//factor), nn.ReLU(), nn.Dropout(cfg.gnn_dropout)])
                dim //= factor

        # Create one node_pooling and then a linear for classification
        self.node_pooling = PoolingLayer(cfg.gnn_node_emb_dim, cfg.gnn_global_hidden_dim)
        self.project = nn.Sequential(*layers)
        self.final_linear = nn.Sequential(
            nn.Linear(dim + cfg.nb_clinical_data_features, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(64, cfg.nb_classes),
        )

    def forward(self, x:torch.Tensor, batch_indexes: torch.Tensor, aux_feats: Optional[torch.Tensor] = None, **kwargs):
        '''
            Params
            ---
            * `x`: Thea batched embedding representation from the gnn: Size: [ |Nodes in batch| x | Node Embedding Dim | ]
            * `batch_indexes`: A tensor with size: | Nodes in Batch | that has 0,1 values based on which nodes appears in what batch
            * `aux_feats`: TODO
        '''
        assert 'lesion_wide_feat_tensor' in kwargs, "No lesion_wide_feat_tensor"
        clinical_data = kwargs['lesion_wide_feat_tensor']
        x_out = self.node_pooling(x, batch_indexes)
        x_out = self.project(x_out)

        # put ffr in the features
        out = torch.hstack([x_out, clinical_data])
        out = self.final_linear(out)
        return out




#
# Multi purpose pooling Layers
#


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
        siamese_mul = 2 if self.is_siamese else 1
        self.siamese_reshaping = in_dim * pooling_mul

        self.lin1= nn.Sequential(
            nn.Linear(in_dim * pooling_mul * siamese_mul, 256),
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
            nn.Linear(16, out_dim)
        )

    def __siamese_batching(self, siamese_ptr, node_embs, batch_indexes):
        ''' Change the batch indexes to pool information for the node embeddings of both graphs. '''
        new_batch_indexes = torch.zeros(node_embs.shape[0], dtype=torch.int64, device=batch_indexes.device)
        curr_idx = 0
        curr_dpoints = 0
        for ptrs in siamese_ptr:
            range_v1_st, range_v1_end = curr_dpoints, ptrs[1] + curr_dpoints
            range_v2_st, range_v2_end = range_v1_end + 1, ptrs[2] + curr_dpoints
            curr_dpoints =  range_v2_end + 1

            new_batch_indexes[range_v1_st:range_v1_end+1] = curr_idx
            curr_idx += 1
            new_batch_indexes[range_v2_st:range_v2_end+1] = curr_idx
            curr_idx += 1

        return new_batch_indexes

    def forward(self, node_embs, batch_indexes, lesion_mask = None, lesion_wide_feat_tensor=None, siamese_ptr=None):
        # lesion masking if requested
        if self.use_lesion_mask:
           node_embs = node_embs[lesion_mask]
           batch_indexes = batch_indexes[lesion_mask]

        # handle siamese
        if self.is_siamese:
            assert siamese_ptr != None
            batch_indexes = self.__siamese_batching(siamese_ptr, node_embs, batch_indexes)


        # node pooling
        out = self.node_pooling(node_embs, batch_indexes)

        # if siamese
        if self.is_siamese:
            # concat siamese vectors (every 2 vectors are siamese)
            out = out.view(-1, 2 * self.siamese_reshaping)

        # projection
        out = self.lin1(out)

        # global info attachment
        if self.gnn_use_global_info:
            out = torch.hstack([out, lesion_wide_feat_tensor])

        # classifier
        return self.lin2(out)


def pooling_resolver(cfg) -> nn.Module:
    if cfg.gnn_pooling_cls_name == "SiamesePooling":
        return SiamesePooling(cfg)
    elif cfg.gnn_pooling_cls_name == "SiamesePoolingWithFFR":
        return SiamesePoolingWithFFR(cfg)
    elif cfg.gnn_pooling_cls_name == "ConfigurablePooling":
        return ConfigurablePooling(cfg)
    else:
        raise RuntimeError(f"We din't have {cfg.gnn_pooling_cls_name} in pooling layers.")


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
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_classes", type=int, default=1, help="The number of classes to predict for our clf task")

    # GNN model parameters
    # -- pooling
    parser.add_argument("--gnn_pooling_cls_name", type=str, default="ConfigurablePooling", help="One of pooling layers define din 'cardio.networks.gnn' ")
    parser.add_argument("--pool_only_lesion_points", action='store_true', help="In case we use whole artery, gnn pooling will pool only lesion points")
    parser.add_argument("--node_pooling", type=str, default='sum', help="How to pool the nodes of the gnn: sum, mean, max, all_stats")
    # -- model
    parser.add_argument("--gnn_is_siamese", default=False, help="In case we use siamese dataset use this flag")
    parser.add_argument("--gnn_cls_name", type=str, default="GIN", help="One of the pred class names in 'torch_geometric.nn.models.basic_gnn' or one of our custom gnn class names, CustomGIN")
    parser.add_argument("--gnn_node_emb_dim", type=int, default=256, help="GNN hidden dim that will be input of the GATconv")
    parser.add_argument("--gnn_global_hidden_dim", type=int, default=512, help="GNN The dimension to have after the pooling ")
    parser.add_argument("--gnn_nb_layers", type=int, default=1, help="GNN number of Transformer layers")
    parser.add_argument("--gnn_dropout", type=float, default=0.415081799988646554, help="GNN The dropout on the Attention layers of the gnn")
    parser.add_argument("--gnn_act", type=str, default="relu", help="The type of activation (one of torch activations)")
    parser.add_argument("--gnn_norm", type=str, default="BatchNorm", help="One of the normalizations, see 'torch_geometric.nn.norm.__init__.py'")
    parser.add_argument("--gnn_freeze_weights", type=bool, default=False, help="Freeze gnn weights (not pooling)")
    parser.add_argument("--gnn_ffr_pooling_factor", type=int, default=1, help="Scale the global dim by a factor multiple times (until dim is >= 16) before you concat it with ffr")
    parser.add_argument("--gnn_ffr_pooling_proj_dim", type=int, default=16, help="Stops scalling when you reach this projection dimention")

    # This is automatically set from the dataset module
    parser.add_argument("--gnn_node_features", type=int, default=3, help="Dim of features in the nodes that we will feed to GNN")

    # CNN model parameters
    parser.add_argument("--standardize_img", action='store_true', help="Standardizes the input of the GNN")
    parser.add_argument("--cnn_dropout", type=float, default=0.45081799988646554, help="The dropout of the CNN fully connected layer at the end")

    # CNN model that inputs data to GNN
    parser.add_argument("--cnn_out_channels", type=int, default=64, help="The num of features the CNN needs to create, and pass to GNN a channels")
    parser.add_argument("--cnn_kernel", type=int, default=3, help="The kernel to use for the CNN")

    # Transformer for radiomix model parameters
    parser.add_argument("--tr_nb_radiomix_features", type=int, default=46, help="The number of different radiomix features we use")
    parser.add_argument("--tr_d_model", type=int, default=512, help="The dim of the transformer embeddings")
    parser.add_argument("--tr_nhead", type=int, default=8, help="The number of heads")
    parser.add_argument("--tr_dropout", type=float, default=0.1, help="The number of heads")
    parser.add_argument("--tr_nb_layers", type=int, default=6, help="The number of layers on the encoder")
    parser.add_argument("--tr_dim_feedforward", type=int, default=2048, help="The feed forward dimension on the encode layers")

    return parser.parse_args()


confs = parse_args()
confs.gnn_nb_layers=12
confs.gnn_dropout=0.435694367117766881
confs.gnn_node_emb_dim=256
confs.node_pooling='mean'
confs.use_lesion_wide_info=True
confs.gnn_pooling_cls_name='ConfigurablePooling'  
confs.lesion_wide_feat_dim=21



model = GNNClassifier(gnn_resolver(confs),
                    pooling_resolver(confs), confs)
print(model)
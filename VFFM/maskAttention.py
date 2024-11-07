import torch 
import numpy as np
import os 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import copy
import torch

def compute_selfattention(transformer_encoder,x,mask,src_key_padding_mask,i_layer,d_model,num_heads):
    h = F.linear(x, transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
    combined_mask = torch.zeros_like(attn_probs)
    if mask is not None:
        combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
    if src_key_padding_mask is not None:
        combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[1],1)
        # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
    combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
    # setting masked logits to -inf before softmax
    attn_probs += combined_mask
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_logits,attn_probs



def extract_selfattention_maps(transformer_encoder,x,mask,src_key_padding_mask):
    attn_logits_maps = []
    attn_probs_maps = []
    num_layers = transformer_encoder.num_layers
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
            attn_logits,attn_probs = compute_selfattention(transformer_encoder,h,mask,src_key_padding_mask,i,d_model,num_heads)
            attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
            attn_probs_maps.append(attn_probs)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attn_logits_maps,attn_probs_maps

# class Fusion (nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

def positional_encoding(max_len,embedding_dim):
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    pos_encoding = torch.zeros((1, max_len, embedding_dim))
    pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
    pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
    return pos_encoding

class SinconPos(nn.Module):
    def __init__(self,max_len,embedding_dim,device) -> None:
        super().__init__()
        self.max_len=max_len
        self.embedding_dim=embedding_dim
        self.position_encod = positional_encoding(max_len,embedding_dim)
        self.device = device

    def forward(self,x):
        # print(x[0])
        # print(self.position_encod[:,:5,:].to(self.device)[0])
        return x+self.position_encod[:,:self.max_len,:].to(self.device)

# learnable position encoding  
class LearnablePositionalEncoding(nn.Module):

    def __init__(self,max_position_embeddings,embed_dim):
        super().__init__()
        self.position_embeddings = nn.parameter.Parameter(torch.zeros(1, max_position_embeddings,embed_dim))

    def forward(self,x):
        position_embeddings = self.position_embeddings

        return x+position_embeddings[:,:5,:]

# fusin model using multi-head attention 
class FusionModule (nn.Module):
    def __init__(self,embedding_dim=1024, dim_feedforward=512,  nhead=4, num_layers=8, patch_dim=8, drop_rate= 0.3,) -> None:
        super().__init__()
        # self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.d_model = embedding_dim
        self.layer_num = num_layers
        self.patch_dim = patch_dim
        self.drop_rate =drop_rate
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True, dim_feedforward= self.dim_feedforward,dropout=self.drop_rate, norm_first=True)
        # self.fusion = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.layer_num)
        self.fusion = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True, dim_feedforward= self.dim_feedforward,dropout=self.drop_rate,norm_first=True)

    def forward(self, x_input):
        # print("device:",x_input.device, mask.device)
        # x= self.fusion(x_input, src_key_padding_mask=mask)
        x= self.fusion(x_input)

        return x

class MAM(nn.Module):
    def __init__(self,feat_dim,att_dim=256, nhead=1, qkv_bias=False, drop_rate=0.2,lesionNum=5, device='cuda'):
        super(MAM,self).__init__()

        self.feat_dim = feat_dim
        # self.proj = nn.Conv1d(in_channels=feat_dim,out_channels=feat_dim,kernel_size=patch_size,stride=patch_size)
        self.learnablePos = LearnablePositionalEncoding(lesionNum,feat_dim)
        # self.sinconPos = SinconPos(max_len=lesionNum,embedding_dim=feat_dim,device=device)
        self.nhead = nhead 
        self.device=device
        # head_dim = dim // nhead

        self.att_dim = att_dim
        self.norm_att = nn.LayerNorm(self.feat_dim)

        self.q = nn.Linear(feat_dim,self.att_dim,bias=qkv_bias)
        self.k = nn.Linear(feat_dim,self.att_dim,bias=qkv_bias)
        self.v = nn.Linear(feat_dim,self.att_dim,bias=qkv_bias)
        # self.att_linear = nn.Linear(att_dim,feat_dim)
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=nhead)
        self.att_norm = nn.LayerNorm(feat_dim)

        self.ff = nn.Sequential(
            nn.Linear(feat_dim, att_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(att_dim, feat_dim),
            nn.Dropout(p=drop_rate),
        )
        self.ff_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, mask):

        x_with_pos=x
        ##########################################
        #pytorch mutihead attention
        ##########################################

        # x = x_with_pos.permute(1, 0, 2)  # Change to (sequence_length, batch_size, input_size) for nn.MultiheadAttention
        # attn_output, _ = self.multihead_attention(x, x, x)
        # attn_output = attn_output.permute(1, 0, 2)  # Change back to (batch_size, sequence_length, input_size)
        # output = self.layer_norm(attn_output + x_with_pos)

        ##########################################
        # mutihead attention python by sun
        ##########################################

        q = self.q(x_with_pos) # batch, patch, att_dim 
        k = self.k(x_with_pos)
        v = self.v(x_with_pos)

        q = q.view(q.size(0), -1, self.nhead, self.att_dim // self.nhead) # batch, patch, nhead, att_dim//nhead
        k = k.view(k.size(0), -1, self.nhead, self.att_dim // self.nhead)
        v = v.view(v.size(0), -1, self.nhead, self.att_dim // self.nhead)

        attn_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.att_dim ** 0.5) # 


        for _, binary_mask in enumerate(mask):
            attn_scores[:,binary_mask==0,]=float('-inf')
        
        attn_probs = F.softmax(attn_scores, dim=1)
        attn_probs = torch.where(torch.isnan(attn_probs), torch.tensor(0.0).to(self.device), attn_probs)


        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1, self.att_dim)   

        attn_output = attn_output + x
        attn_output = self.att_norm(attn_output)

        #feed forward 
        ffd = self.ff (attn_output)+attn_output

        output = self.ff_norm(ffd)

        return output
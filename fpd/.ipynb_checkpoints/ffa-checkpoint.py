import sys
from torch import nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS
from mmcv.runner import BaseModule
import numpy as np
import math
import torch
import torch.nn.functional as F


@AGGREGATORS.register_module()
class PrototypesDistillation(BaseModule):
    def __init__(self, num_queries, dim, num_base_cls=15, num_novel=0):
        super().__init__()
        self.num_queries = num_queries
        self.num_base_cls = num_base_cls
        self.num_novel = num_novel
        self.dim = dim

        k_dim = dim // 4
        self.k_dim = k_dim

        self.query_embed = nn.Embedding(num_queries*num_base_cls, k_dim)
        self.duplicated = False
        if self.num_novel > 0:
            self.query_embed_novel = nn.Embedding(num_queries * num_novel, k_dim)
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

    def forward(self, support_feats, support_gt_labels=None, forward_novel=False, forward_novel_test=False):
        """
        Args:
            support_feats: Tensor with shape (B, C, H, W).
            support_gt_labels: Support gt labels.
            forward_novel (bool): Novel classes.
            forward_novel_test (bool): Test time.
        Returns:
            tensor with shape (15, 1024)
        """
        # at the fine-tuning stage, duplicate the most compatible feature queries for the novel classes
        # ************************************************************
        if not self.duplicated and self.num_novel > 0 and not forward_novel_test and forward_novel:
            with torch.no_grad():
                # support_gt_labels = torch.tensor(list(dict.fromkeys(support_gt_labels.tolist())), device='cuda:0')
                # selected_indices = []
                # for i in range(0, 10, 2):
                #     a = []
                #     a.append(i)
                #     a.append(i + 1)
                #     selected_indices.append(np.random.choice(a))
                # support_feats = support_feats[selected_indices]

                support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
                B, C, H, W = support_feats_mp.shape
                k = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
                k = self.w_qk(k)  # (B, 196, 1024)
                query_emb = self.query_embed.weight
                q = query_emb.unsqueeze(0).repeat(B, 1, 1)
                B, Nt, E = q.shape
                attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
                weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)
                drop = 5
                top_indices = torch.topk(weight, self.num_queries + drop, dim=-1)[1][:, -self.num_queries:]
                top_emb = torch.gather(self.query_embed.weight.unsqueeze(0).expand(B, -1, -1), 1, top_indices.unsqueeze(-1).expand(-1, -1, self.k_dim))
                top_emb = top_emb[torch.sort(support_gt_labels, dim=0)[1]].reshape(self.num_novel*self.num_queries, self.k_dim)
                self.query_embed_novel.weight.copy_(top_emb)
            self.duplicated = True  # set as True once duplicated
        # ************************************************************

        support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
        B, C, H, W = support_feats_mp.shape
        k = v = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
        k = self.w_qk(k)  # (B, 196, 1024)

        # scaled dot-product attention
        if forward_novel:
            query_emb = self.query_embed_novel.weight
            q = query_emb.reshape(self.num_novel, self.num_queries, query_emb.size(-1))
        else:
            query_emb = self.query_embed.weight
            q = query_emb.reshape(self.num_base_cls, self.num_queries, query_emb.size(-1))
        q = q[support_gt_labels, ...]  # align with support_gt_labels
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
        weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)
        prototypes = torch.matmul(attn.softmax(-1), v)     # (B, 5, 1024)
        return weight, prototypes

        # attention heatmap of feature queries upon support images 热力图
#     if len(support_gt_labels) == 0:
#         pass
#     else:
#         import cv2
#         import os
#         os.makedirs('attention_heatmap', exist_ok=True)
#         if not support_feats.size(0):
#             sys.exit()
#         if img_metas is not None:
#             img = img_metas[-1]
#         bs = len(support_gt_labels)
#         prefix = 'novel' if forward_novel else 'base'
#         for img_id in range(bs):
#             gt_label = support_gt_labels[img_id].cpu().numpy()
#             file_name = img_metas[img_id]['filename'].split('/')[-1]
    
#             # # attn heat map
#             attn2 = attn.squeeze(1).softmax(-1)  # (16, 5, 49)
#             attn2 = (attn2 - attn2.min(dim=2, keepdim=True)[0]) / (attn2.max(dim=2, keepdim=True)[0] - attn2.min(dim=2, keepdim=True)[0])
#             for q_id in range(attn2.size(1)):
#                 attn_hm = attn2[img_id, q_id:q_id+1, :].reshape(1, 1, support_feats_mp.size(-2), support_feats_mp.size(-1))
#                 attn_hm = F.interpolate(attn_hm, size=(224, 224), mode='bilinear', align_corners=True)[0].permute(1, 2, 0).cpu().numpy()
#                 mean = torch.ones((3, 224, 224)) * torch.tensor([103.530, 116.280, 123.675])[:, None, None]
#                 raw_im = img[img_id, :3].add(mean.cuda()).permute(1, 2, 0).detach().cpu().numpy()
                
#                 heatmap = cv2.applyColorMap((attn_hm * 255).astype('uint8'), cv2.COLORMAP_JET)
#                 result = cv2.addWeighted(raw_im.astype('uint8'), 0.6, heatmap, 0.4, 0)
#                 cv2.imwrite(f'attention_heatmap/{prefix}_class{gt_label}_{file_name.split(".")[0]}_query{q_id}.png', result)
        

@AGGREGATORS.register_module()
class PrototypesAssignment(BaseModule):
    def __init__(self, dim, num_bg=5):
        super().__init__()
        k_dim = dim // 4
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

        self.num_bg = num_bg
        if self.num_bg > 0:
            self.dummy = nn.Parameter(torch.Tensor(self.num_bg, dim))
            nn.init.normal_(self.dummy)
            self.linear = nn.Linear(dim, k_dim)
        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, query_feature, prototypes, query_img_metas=None):
        """
        Args:
            query_feature: Tensor with shape (B, C, H, W)
            prototypes: Tensor with shape (num_supp, num_queries, C),
            query_img_metas: Visualization.
        Returns:
            class-specific query feature: tensor(B, C, H, W)
        """

        B, C, H, W = query_feature.shape
        num_supp, num_queries, _ = prototypes.shape
        q = query_feature.reshape(B, C, H*W).permute(0, 2, 1)   # (B, H*W, 1024)
        k = v = prototypes.reshape(num_supp * num_queries, C)

        q = self.w_qk(q)
        k = self.w_qk(k)

        if self.num_bg > 0:
            dummy_v = torch.zeros((self.num_bg, C), device='cuda')
            k = torch.cat([k, self.linear(self.dummy)], dim=0)
            v = torch.cat([v, dummy_v], dim=0)

        k = k.unsqueeze(0)
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.expand(B, -1, -1).transpose(-2, -1))
        attn.div_(0.5)

        out = torch.matmul(attn.softmax(-1), v)  # (B, 2850, 1024)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = query_feature + self.gamma * out
        return out

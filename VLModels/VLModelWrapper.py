"""
basically, we have LXMERT and VLBERT
1. do they have the same APIs
2. do they have the same tokenizer
3. do they have the same seq_length, incuding token_len and box_len
4. do they have same feat_dim
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------------------------------- #
# ---------------------- VLBERT ----------------------- #
# ----------------------------------------------------- #
from ProjUtils.Constant import VLBERTConfigFile
from VLModels.VLBERT.pretrain.function.config import config as vlbert_config_from_edict
from VLModels.VLBERT.pretrain.function.config import update_config
from VLModels.VLBERT.common.visual_linguistic_bert import VisualLinguisticBertForPretraining


import sys
import socket
HOST = socket.gethostname()
if HOST.startswith("t"):
    sys.path.append('/tank/space/xugy07/MetaVLScratch')
elif HOST.startswith("a"):
    sys.path.append('/home/xu/MetaVL')


class VLBERTModel(nn.Module):
    """
    ICLR's VLBERT paper has more complex implementation
    1. add many embeddings
    2. data_format: [txt_token, vis_token, pad]
    """
    def __init__(self):
        """
        no params
        1. dict with default values as config
        2. using base_e2e_16x16G_fp16.yaml to update certain fields
        """
        super(VLBERTModel, self).__init__()

        # -----------------------
        # the logic is:
        # 1. vlbert has its own config: _C = edict(), easy dict
        #   1.1 easy_dict already has all things.
        #   1.2 using VLEBRTConfigFile to update eady_dict
        # 2. core code:
        #   2.1 _C = edict()
        #   2.2 config = _C
        # therefore, we dont need the config in init funciton.
        # -----------------------
        update_config(VLBERTConfigFile)
        self.vlbert_cfg = vlbert_config_from_edict


        # --------------------
        # model dfinition
        # BaseModel (Just ignore it, it just like nn.module)
        #   VisualLinguisticBert
        #       VisualLinguisticBertForPretraining
        # --------------------
        # 1. Visual Embedding Part
        self.bbox_txt_embedding = nn.Embedding(1, self.vlbert_cfg.NETWORK.VLBERT.hidden_size)
        self.bbox_linear_proj = nn.Linear(2048, self.vlbert_cfg.NETWORK.VLBERT.visual_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Encoder Part
        network_config = self.vlbert_cfg.NETWORK.VLBERT
        self.vlert_with_cpt_pred_head = VisualLinguisticBertForPretraining(
            network_config,
            with_rel_head=False,
            with_mvrc_head=False,
            with_mlm_head=True
        )

        # 4. loss_func
        self.mlm_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.contrastive_align_loss = None


    def get_txt_visual_token(self, txt_span_idx, bbox_resnet_feats):
        """
        for each txt_token, we have the whole image bbox
        1. img_feat, bbox1, bbox2 ...
        2. txt should have img_feat information.
        """
        # ---------------------------
        # range: 0 --> batch size
        # add another dimension using [:,None]
        # ---------------------------
        row_id = txt_span_idx.new_zeros(txt_span_idx.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]    # [[0], [1] ... [7]]
        row_id += row_id_broadcaster

        # --------------------
        # extract feats:
        # 1. row_id.view(-1): 32 * 64 --> 2048
        # 2. text_span_tags.view(-1): 32 * 64 --> 2048
        # bbox_resnet_feats: 32 100 2048
        # --------------------
        # index, we can use as many times as possible
        # 1. index is tuple [0,0] [1,2]...
        # --------------------
        return bbox_resnet_feats[row_id.view(-1), txt_span_idx.view(-1)].view(*txt_span_idx.shape, -1)



    def forward(self,
                bbox_resnet_feats,
                bbox_position_feats,  # not used for feature input, but used to get bbox_mask
                txt_token_ids,
                txt_token_labels,
                reduce=True):
        """
        visual input:
            1. resnet feat from Detectron2
            2. coordinate pos information
        text input:
            1. ids after tokenizer
            2. txt lens
        Output:
            true lables: comp_cpt
        """
        # --------------------------
        # deal with bbox
        # --------------------------
        batch_size, bbox_num = bbox_resnet_feats.shape[0], bbox_resnet_feats.shape[1]
        bbox_resnet_feats = self.bbox_linear_proj(bbox_resnet_feats)
        bbox_linguistic_token = self.bbox_txt_embedding(bbox_resnet_feats.new_zeros((batch_size, bbox_num)).long())
        # concat [bbox_feat, [IMG]], not using the positional feats
        bbox_concat_vl_emb = torch.cat((bbox_resnet_feats, bbox_linguistic_token), -1)
        # bbox mask
        bbox_mask = bbox_position_feats.sum(-1) > 1e-6  # how to get the bbox feats

        # --------------------------
        # deal with text
        # --------------------------
        txt_token_type = txt_token_ids.new_zeros(txt_token_ids.shape)     # 32 * 64
        txt_token_mask = (txt_token_ids > 0)
        txt_span_idx = txt_token_ids.new_zeros(txt_token_ids.shape)
        txt_visual_token = self.get_txt_visual_token(txt_span_idx, bbox_resnet_feats) # 32 64 2048

        # ----------------------------------------------------------------------------------------------------
        # Different ways to organize the txt/visual input
        # 1. vl_embeddings[grid_pos < text_end] = text_vl_embeddings[text_mask]
        # 2. vl_embeddings[(grid_pos >= text_end) & (grid_pos < bbox_end)]  = bbox_vl_embeddings[bbox_mask]
        # ---------------------------------------------------------------------------------------------------
        relationship_logits, cpt_token_logit, mvrc_logits = self.vlert_with_cpt_pred_head(
            txt_token_ids,
            txt_token_type,
            txt_visual_token,
            txt_token_mask,
            bbox_concat_vl_emb,
            bbox_mask
        )

        # -----------------------------
        # self-attn based on batch infor to update parameters:
        # 1. max_length
        # 2. max_length_for_current_batch
        # -----------------------------
        # based on batch to do calculation:
        # 1. cpt_token_logit:
        # 2. true_token_label:
        # -----------------------------
        if cpt_token_logit.size(1) < txt_token_labels.size(1):
            txt_token_labels = txt_token_labels[:, :cpt_token_logit.size(1)]
        else:
            cpt_token_logit = cpt_token_logit[:, :txt_token_labels.size(1)]

        # -------------------
        # reduce or not
        # -------------------
        if reduce:
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        loss = loss_func(cpt_token_logit.contiguous().view(-1, cpt_token_logit.size(-1)),
                         txt_token_labels.contiguous().view(-1))
        return {
            'loss': loss,
            'score': cpt_token_logit,
        }

# ----------------------------------------------------- #
# ---------------------- LXMERT ----------------------- #
# ----------------------------------------------------- #

from VLModels.LXMERT.lxrt.modeling import LXRTPretraining, BertConfig
class LXMERTModel(nn.Module):
    def __init__(self):
        """
        two types of cfgs: meta_cfg, vl_cfg
        """
        super(LXMERTModel, self).__init__()

        # 1. BertConfig
        self.lxmert_cfg = BertConfig(
            vocab_size_or_config_json_file = 30522,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=3,
            initializer_range=0.02
        )

        # 2. define the model
        self.lxmert_pretrain_model = LXRTPretraining(
            self.lxmert_cfg,
            task_mask_lm=True,
            task_obj_predict=False,
            task_matched=False,
            task_qa=False,
            visual_losses='obj,attr,feat',
            num_answers=2
        )

        # 3. train from scratch: already done LXRTPretraining
        # self.model.apply(self.model.init_bert_weights)



    def forward(self,
                bbox_resnet_feats,
                bbox_position_feats,
                txt_token_ids,
                txt_token_labels,
                text_len = None,  # this is for VLBERT
                reduce=True,
                **kwargs):
        """
        1. Visual Input:
            1.1 rest_feats
            1.2 4-Dim posotion feats
        2. Text Input:
            2.1 token_id
            2.2 token_label
        actually, we don't need the text_len in lxmert
        """
        # --------------------------------
        # further enhance the txt input
        # --------------------------------
        # only having one sent
        txt_token_types = txt_token_ids.new_zeros(txt_token_ids.shape)
        # Bert's [mask] ID = 0
        txt_token_mask = (txt_token_ids > 0)

        # --------------------------------------------------------------------------
        # mlm_logits: batch_size * seq_length * vocab_size
        # masked_token_bert_emb: masked_token_num  * 384, like 67 (> 32 * 2) * 384
        # --------------------------------------------------------------------------
        loss_scalar_for_bp, loss_tup_for_diff_loss_stat, mlm_logits, txt_token_emb, masked_token_pos = self.lxmert_pretrain_model(
            txt_token_ids,
            txt_token_types,
            txt_token_mask,
            txt_token_labels,
            bbox_resnet_feats,
            bbox_position_feats,
            None, None, None,
            reduce=reduce
        )


        # returned results
        return {
            'loss': loss_scalar_for_bp,
            'score': mlm_logits,
            'txt_token_emb': txt_token_emb,
            'mask_token_pos': masked_token_pos
        }





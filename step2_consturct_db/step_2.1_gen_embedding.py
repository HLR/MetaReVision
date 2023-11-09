"""
we can refer validate file


decision choice:
1. which model to load
---------------------
best_avg_val_acc_model.pth
best_avg_val_loss_model.pth
best_avg_val_ppl_model.pth
----------------------



pipeline:
1. load ckpt
2. design data structure

"""
import os
import argparse
import tqdm
from yacs.config import CfgNode
import stanza
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader

from VLModels.VLModelWrapper import VLBERTModel, LXMERTModel
from DataSet.BatchCollator import BatchCollator
from DataSet.NormalDataSet import MSCOCONormalDataSet, FlickrNormalDataSet
from ProjUtils.Constant import MetaExpList
from ProjUtils.MetaTrainUtils import add_base_model_dir_into_cfgnode
from ProjUtils.CkptUtils import CheckpointerFromCfg


# -----------------------
# global setting
# -----------------------
# device and tools
device = 'cuda'

# machine
import sys, socket
HOST = socket.gethostname()
if HOST.startswith("t"):
    sys.path.append('/tank/space/xugy07/MetaVLScratch')
    SupCfgFile  = '/tank/space/xugy07/MetaVLScratch/Config/supervise_cfg.yaml'
    MetaCfgFile = '/tank/space/xugy07/MetaVLScratch/Config/maml_cfg.yaml'
    OutDir = '/tank/space/xugy07/MetaVLScratch/runs'
elif HOST.startswith("a"):
    sys.path.append('/home/xu/MetaVL')
    SupCfgFile = '/home/xu/MetaVL/Config/supervise_cfg.yaml'
    MetaCfgFile = '/home/xu/MetaVL/Config/maml_cfg.yaml'
    OutDir = '/home/xu/MetaVL/runs'
elif HOST.startswith("l"):
    sys.path.append('/localscratch2/xugy/MetaCompLearn')
    SupCfgFile = '/localscratch2/xugy/MetaCompLearn/Config/supervise_cfg.yaml'
    MetaCfgFile = '/localscratch2/xugy/MetaCompLearn/Config/maml_cfg.yaml'
    OutDir = '/localscratch2/xugy/MetaCompLearn/runs'


# ------------------
# ckpt mode list
# ------------------
CKPTModelList = [
    # ------------ novel ckpt ----------- #
    # "novel_best_avg_val_loss_model.pth",
    "novel_best_avg_val_acc_model.pth",
    # "novel_best_pair_acc_model.pth",
    #
    # ------------ seen ckpt ----------- #
    # "seen_best_avg_val_acc_model.pth",
    # "seen_best_avg_val_loss_model.pth",
    # "seen_best_avg_val_ppl_model.pth",
    #
    # ------------ both ckpt ----------- #
    #"best_avg_val_acc_model.pth",
    #"novel_avg_val_loss_model.pth",
    #"best_avg_val_ppl_model.pth",
    #
]


# ----------------
# DB Item
# ----------------
class DBValueItem(object):
    def __init__(self, element_cpt_vec, element_cpt_str, dictItem, element_type):
        self.element_cpt_vec = element_cpt_vec
        self.element_cpt_str = element_cpt_str
        self.dictItem = dictItem
        self.element_type = element_type


def gen_emb_db(data_loader, model):
    """
    data_loader: with batch_size = 1
    model:
    """
    # result
    state_vec_list = []
    obj_vec_list = []
    cnt = 0

    # access each item
    for dictItem in tqdm.tqdm(data_loader):
        # --------------------
        # batch_soize == 1:
        #   'img_id' = {Tensor: 1} tensor([577196])
        #   'pair' = {list: 1} ['wearing_man']
        #   'attr' = {list: 1} ['wearing']
        #   'obj' = {list: 1} ['man']
        #   'state_pos' = {list: 1} [[4]]
        #   'obj_pos' = {list: 1} [[2]]
        #   'mask_cap' = {Tensor: 1*64} tensor([[ 101, 1996,  103, 2003 ...]
        #   'mask_label' = {Tensor: 1*64} tensor([[  -1,   -1, 2158, ...]
        #   'bbox_feats' = {Tensor: 1*100*2048}
        #   'bboxes' = {Tensor: 1*100*4}
        #   'bbox_num' = {Tensor: 1*1}
        # [mask]: 103 --> [label]: 2158
        # --------------------
        img_id = dictItem['img_id']
        # captions
        txt_token_ids = dictItem['mask_cap']
        txt_token_labels = dictItem['mask_label']
        # img feats
        bbox_resnet_feats = dictItem['bbox_feats']
        bbox_position_feats = dictItem['bboxes']
        # state/obj pos
        state_pos = [item for sublist in dictItem["state_pos"] for item in sublist]
        obj_pos = [item for sublist in dictItem["obj_pos"] for item in sublist]
        element_cpt_pos = state_pos + obj_pos
        state_pos, obj_pos = torch.tensor(state_pos), torch.tensor(obj_pos)

        # ------------------
        # cpu --> gpu
        # ------------------
        txt_token_labels = txt_token_labels.to(device)
        txt_token_ids = txt_token_ids.to(device)
        bbox_resnet_feats = bbox_resnet_feats.to(device)
        bbox_position_feats = bbox_position_feats.to(device)

        # -----------------
        # go through model
        # 'loss' = {Tensor: 64} ([0.0000, 0.0000, 0.3716, 0.0110, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
        # 'score' = {Tensor: 1*64*30522}
        # 'txt_token_emb' = {Tensor: 1*64*384}
        # 'mask_token_pos' = {Tensor: 1*64} tensor([[False, False,  True,  True, ...
        # -----------------
        dictResult = model(bbox_resnet_feats=bbox_resnet_feats,
                           bbox_position_feats=bbox_position_feats,
                           txt_token_ids=txt_token_ids,
                           txt_token_labels=txt_token_labels,
                           reduce = False)

        # ------------------
        # checking here
        # ------------------
        # 2D_pos -> 1D_pos
        extract_cpt_pos = dictResult['mask_token_pos'].squeeze().nonzero().reshape(-1).tolist()
        assert set(extract_cpt_pos) == set(element_cpt_pos), "extracted cpt should match"
        # extrac state/obj
        state_emb = dictResult['txt_token_emb'].squeeze(0)[state_pos].mean(0)
        obj_emb = dictResult['txt_token_emb'].squeeze(0)[obj_pos].mean(0)

        # -----------------
        # DB key item
        # -----------------
        obj_vec_list.append(obj_emb)
        state_vec_list.append(state_emb)

        # -----------------
        # DB value item
        # -----------------
        #attr_db_item = DBValueItem(
        #    element_cpt_vec = state_emb.cpu().numpy(),
        #    element_cpt_str = ' '.join(dictItem['attr']),
        #    dictItem = dictItem,
        #    element_type='attr'
        #)

        #obj_db_item = DBValueItem(
        #    element_cpt_vec = obj_emb.cpu().numpy(),
        #    element_cpt_str = ' '.join(dictItem['obj']),
        #    dictItem = dictItem,
        #    element_type='obj'
        #)

        # -----------------------------
        # check tokenizer
        # [CLS] a [MASK] [MASK] that is parked in front of other buses. [SEP] [PAD] [PAD]  .... [PAD]
        # -----------------------------
        # txt_token = dictItem['mask_cap']
        # tokenizer = data_loader.dataset.bert_tokenizer
        # print(tokenizer.decode(txt_token.squeeze().tolist()))


    # -----------------------------------
    # not put into DB just save to pkl
    # 1. key vec to pickle
    # 2. we already have item list
    # -----------------------------------
    state_vec_datastore = torch.stack(state_vec_list)
    obj_vec_datastore = torch.stack(obj_vec_list)

    torch.save(state_vec_datastore, 'state_datastore.pt')
    torch.save(obj_vec_datastore, 'obj_datastore.pt')


# ---------------
# main
# ---------------
if __name__ == "__main__":
    # -------------------
    # config part
    # -------------------
    # config from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='lxmert')
    parser.add_argument('--dataset', type=str, default='mscoco')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--novel_comps', action='store_true')
    parser.add_argument('--do_analysis', type=str, default='True')
    args = parser.parse_args()

    # fixed config
    args.exp_type = "supervised"

    # config from cfgNode
    if args.exp_type.startswith("super"):
        args.config_file = SupCfgFile
    elif args.exp_type in MetaExpList:
        args.config_file = MetaCfgFile
    cfg_node = CfgNode(new_allowed=True)            # {}
    cfg_node.merge_from_file(args.config_file)

    # modify cfgNode from cmd
    cfg_node.base_model = args.base_model
    cfg_node.dataset = args.dataset
    cfg_node.seed = args.seed
    cfg_node.output_dir = OutDir
    cfg_node.is_novel_comps = args.novel_comps
    cfg_node.exp_type = args.exp_type
    cfg_node.val.val_item_num = cfg_node.val.batch_size * 100
    cfg_node.do_analysis = args.do_analysis

    # additional config
    add_base_model_dir_into_cfgnode(cfg_node)

    # ---------------------
    # load base model
    # ---------------------
    if cfg_node.base_model == 'vlbert':
        base_model = VLBERTModel()
    elif cfg_node.base_model == 'lxmert':
        base_model = LXMERTModel()
    else:
        raise NotImplementedError('Unknown model type {}'.format(cfg_node.base_model))
    base_model.to(device)

    # -----------------------------------
    # dataloader with BatchSize = 1
    # -----------------------------------
    # data_tools
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/localscratch2/xugy/CacheDir')
    nlp_stanza = stanza.Pipeline('en', dir='/localscratch2/xugy/CacheDir', download_method=None)
    batch_collator = BatchCollator(cfg_node.dataset)
    # data_set
    if cfg_node.dataset.lower().startswith('m'):
        train_ds = MSCOCONormalDataSet(cfg_node, "train", bert_tokenizer, nlp_stanza)
    elif cfg_node.dataset.lower().startswith('f'):
        train_ds = FlickrNormalDataSet(cfg_node, "train", bert_tokenizer,  nlp_stanza)
    # data_loader
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=batch_collator)

    # ------------------------
    # Loading models:
    #   dir: '/egr/research-hlr/xugy07/MetaCompExperiment/flickr/supervised/lxmert_epoch_15_lr_0.0001'
    #   ckpt_file: 'novel_best_avg_val_acc_model.pth'
    # ------------------------
    target_model_dir = cfg_node.base_model_dir
    for ckpt_file in CKPTModelList:
        target_model_file = os.path.join(target_model_dir, ckpt_file)
        ckpt_type = ckpt_file.split('.')[0]


        # ------------------
        # cfg,
        # model,
        # optimizer=None,
        # scheduler=None,
        # save_dir="",
        # save_to_disk=True,
        # logger=None,
        # ------------------
        checkpointer = CheckpointerFromCfg(
            cfg_node,
            base_model,
            save_dir=target_model_dir,
        )

        checkpointer = checkpointer.load(target_model_file, use_latest=False)

        # setting to eval mode
        base_model.to(device).eval()
        with torch.no_grad():
            gen_emb_db(train_loader, base_model)
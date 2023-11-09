"""
refer to what?
1. supervised training
2. which project?

------------------------------------


"""

# -------------------
# for cmd running
# -------------------
import sys
import socket
HOST = socket.gethostname()
if HOST.startswith("t"):
    sys.path.append('/tank/space/xugy07/MetaVLScratch')
elif HOST.startswith("a"):
    sys.path.append('/home/xu/MetaVL')


import os
# config
import argparse
from yacs.config import CfgNode
# tools
from transformers import BertTokenizer
import stanza
# utils
from ProjUtils.ConfigUtils import save_cfg_node
from ProjUtils.SeedUtils import fix_seed
from ProjUtils.FileUtils import mkdir
from ProjUtils.CkptUtils import CheckpointerFromCfg
from ProjUtils.Constant import ProjDir
# model
from VLModels.VLModelWrapper import VLBERTModel, LXMERTModel
from pytorch_transformers import AdamW
# data
from torch.utils.data import DataLoader
from DataSet.BatchCollator import BatchCollator
from DataSet.NormalDataSet import MSCOCONormalDataSet, FlickrNormalDataSet
# trainer
import Trainer.NormalTrainer.SuperTrainer as SuperviseTrainer
from ProjUtils.MetaTrainUtils import add_base_model_dir_into_cfgnode
# contant
from ProjUtils.Constant import ProjDir
# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# wandb
# os.environ["WANDB_SILENT"] = "true"


def train(cfg_node):
    # --------------
    # global device
    # --------------
    device = 'cuda'

    # ------------------------
    # Init Base VLModel
    # ------------------------
    if cfg_node.base_model == 'vlbert':
        # vlbret has its own config
        base_vl_model = VLBERTModel()
    elif cfg_node.base_model == 'lxmert':
        base_vl_model = LXMERTModel()
    else:
        raise NotImplementedError('Unknown model type {}'.format(cfg_node.base_model))
    base_vl_model.to(device)

    # ---------------
    # Optimizer
    # ---------------
    optimizer = AdamW(filter(lambda x: x.requires_grad, base_vl_model.parameters()),
                      lr=cfg_node.train.lr, correct_bias=False, eps=1e-4)


    # -----------------
    # shared tools
    # -----------------
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/localscratch2/xugy/CacheDir')
    nlp_stanza = stanza.Pipeline('en', dir='/localscratch2/xugy/CacheDir')

    # --------------------------------------
    # DataSet and DataLoader
    # --------------------------------------
    batch_collator = BatchCollator(cfg_node.dataset)
    if cfg_node.dataset.lower().startswith('m'):
        # data set
        train_ds = MSCOCONormalDataSet(cfg_node, "Train", bert_tokenizer, nlp_stanza)
        val_seen_ds = MSCOCONormalDataSet(cfg_node, "Val_Seen", bert_tokenizer, nlp_stanza)  # capitalized to match file name
        val_novel_ds = MSCOCONormalDataSet(cfg_node, "Val_Novel", bert_tokenizer, nlp_stanza)
        val_ds = MSCOCONormalDataSet(cfg_node, "Val", bert_tokenizer, nlp_stanza)
    elif cfg_node.dataset.lower().startswith('f'):
        # data set
        train_ds = FlickrNormalDataSet(cfg_node, "Train", bert_tokenizer, nlp_stanza)
        val_seen_ds = FlickrNormalDataSet(cfg_node, "Val_Seen", bert_tokenizer, nlp_stanza)   # capitalized to match file name
        val_novel_ds = FlickrNormalDataSet(cfg_node, "Val_Novel", bert_tokenizer, nlp_stanza)
        val_ds = FlickrNormalDataSet(cfg_node, "Val", bert_tokenizer, nlp_stanza)
        # data loader
    else:
        raise ValueError('unknown DataSet {}'.format(cfg_node.dataset))

    train_data_loader = DataLoader(train_ds, batch_size=cfg_node.train.batch_size, shuffle=True, collate_fn=batch_collator)
    val_seen_dataloader = DataLoader(val_seen_ds, batch_size=cfg_node.val.batch_size, shuffle=False, collate_fn=batch_collator)
    val_novel_dataloader = DataLoader(val_novel_ds, batch_size=cfg_node.val.batch_size, shuffle=False, collate_fn=batch_collator)
    val_dataloader = DataLoader(val_ds, batch_size=cfg_node.val.batch_size, shuffle=False, collate_fn=batch_collator)

    # -----------------------------
    # prepare training enviroment
    # -----------------------------
    # 1. init train_statics
    dict_TrainStats = {
        "global_batch_step": -1,
        "global_epoch_step": 0
    }

    # 2. ckpt
    checkpointer = CheckpointerFromCfg(
        cfg_node,
        base_vl_model,
        optimizer,
        None,
        cfg_node.base_model_dir,
        save_to_disk=True
    )

    # 3. Training Process3
    for epoch_idx in range(cfg_node.train.epoch_num):
        dict_TrainStats["global_epoch_step"] = epoch_idx
        SuperviseTrainer.train_one_epoch(cfg_node,
                                         base_vl_model,
                                         optimizer,
                                         train_data_loader,
                                         val_seen_dataloader,
                                         val_novel_dataloader,
                                         val_dataloader,
                                         checkpointer,
                                         device,
                                         dict_TrainStats)

        # ---------------------------------------------------------------
        # Save ckpt every epoch
        # 1. dict_TrainArgument is really a dict, then we can pass this dict
        # 2. 'model_{:02d}'.format(epoch) is the file name
        # so we only base on epoch to save model
        # ---------------------------------------------------------------
        # checkpointer.save('model_at_epoch_{}'.format(epoch_idx), **dict_TrainArgs)
    return base_vl_model



def main(args):

    # ---------------------------------
    # init cfg_node using yaml file
    # ---------------------------------
    cfg_node = CfgNode(new_allowed=True)        # null node
    cfg_node.merge_from_file(args.config_file)

    # --------------------------
    # modify cfg_node using args
    # --------------------------
    cfg_node.base_model = args.base_model
    cfg_node.dataset = args.dataset
    cfg_node.seed = args.seed
    cfg_node.output_dir = args.output_dir
    cfg_node.is_novel_comps = args.novel_comps
    cfg_node.exp_type = args.exp_type
    cfg_node.val.val_item_num = cfg_node.val.batch_size * 100   # 6400
    cfg_node.version = args.version

    # ---------------------------------------
    # modify cfg_node using constant values
    # ---------------------------------------
    cfg_node.running_mode = 'train'

    # ------------------------------------------------
    # output_dir: run/dataset/exp_type/time_stamp
    # ------------------------------------------------
    # cfg_node.output_dir = os.path.join(cfg_node.output_dir,
    #                                   cfg_node.dataset,
    #                                   cfg_node.exp_type,
    #                                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    """
    exp_config = "{}_epoch_{}_lr_{}".format(cfg_node.base_model, cfg_node.train.epoch_num,  cfg_node.train.lr)
    cfg_node.output_dir = os.path.join(cfg_node.output_dir,
                                       cfg_node.dataset,
                                       cfg_node.exp_type,
                                       exp_config)
    mkdir(cfg_node.output_dir)
    """
    add_base_model_dir_into_cfgnode(cfg_node)


    # --------------------------
    # print config
    # --------------------------
    logger.info("Running with config:\n{}".format(cfg_node))

    # -----------------------------
    # save config to output_dir
    # -----------------------------
    output_config_file = os.path.join(cfg_node.output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_file))
    save_cfg_node(cfg_node, output_config_file)


    # --------------------------
    # train func
    # --------------------------
    train(cfg_node)



if __name__ == "__main__":
    """
    1. cfg
    2. dataset -- dataloader
    3. trainer:
        3.1 gradient
        3.2 optimizer
        3.3 update
    """

    # -----------------------
    # argparse from cmd
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='lxmert')
    parser.add_argument('--dataset', type=str, default='mscoco')
    parser.add_argument('--config_file', type=str, default='{}/Config/supervise_cfg.yaml'.format(ProjDir))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument('--output_dir', type=str, default='{}/runs'.format(ProjDir))
    parser.add_argument('--novel_comps', action='store_true')
    parser.add_argument('--exp_type', choices=["ground", "supervise", "maml", "fomaml", "reptile"], default="supervised")
    parser.add_argument('--version', type=int, default=1)
    args = parser.parse_args()

    # -----------------------
    # fix seed
    # -----------------------
    fix_seed(args.seed)

    # -----------------------
    # main func
    # -----------------------
    main(args)
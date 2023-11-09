"""
current target:
1. using the wandb to sweep model parameters

-----------------------------------------------------

after using sweep:
https://github.com/wandb/client/issues/982

"""

# -------------------
# for cmd running
# -------------------
import sys
import socket
HOST = socket.gethostname()
if HOST.startswith("t"):
    sys.path.append('/tank/space/xugy07/MetaVLScratch')
    cfg_file = '/tank/space/xugy07/MetaVLScratch/Config/maml_cfg.yaml'
    # out_dir = '/tank/space/xugy07/MetaVLScratch/runs'
    out_dir = "/egr/research-hlr/xugy07/MetaCompExperiment"
elif HOST.startswith("a"):
    sys.path.append('/home/xu/MetaVL')
    cfg_file = '/home/xu/MetaVL/Config/maml_cfg.yaml'
    out_dir = '/home/xu/MetaVL/runs'



import os
from time import gmtime, strftime
import json
# config
import argparse
from yacs.config import CfgNode
# tools
from transformers import BertTokenizer
import stanza
# utils
from ProjUtils.MetaTrainUtils import add_episode_data_dir_into_cfgnode, add_meta_exp_dir_into_cfgnode, \
    str2bool, add_base_model_dir_into_cfgnode
from ProjUtils.ConfigUtils import save_cfg_node
from ProjUtils.SeedUtils import fix_seed
from ProjUtils.WandbUtils import convert_cfgnode_to_dict
# model
from VLModels.VLModelWrapper import VLBERTModel, LXMERTModel
# data
# trainer
# from Trainer.NormalTrainer.MAMLTrainer import MAMLTrainer
from DataPreprocessor.StepN_TaskGenerator.FaissMAMLTrainer import FaissMAMLTrainer
from ProjUtils.Constant import MetaExpList, PureMetaExpList, ProjDir, ExpDir
# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from DataPreprocessor.StepN_TaskGenerator.mscoco_transform_json_to_tensor import OneEpisodeDataSet

# ------------------   wandb  ------------------------#
import wandb
# ----------------------------------------------------#

def meta_train(cfg_node):
    # --------------
    # global device
    # --------------
    device = 'cuda'

    # -----------------
    # shared tools
    # -----------------
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp_stanza = stanza.Pipeline('en')

    # ------------------------
    # Init Base VLModel
    # ------------------------
    if cfg_node.base_model == 'vlbert':
        base_vl_model = VLBERTModel()
    elif cfg_node.base_model == 'lxmert':
        base_vl_model = LXMERTModel()
    else:
        raise NotImplementedError('Unknown model type {}'.format(cfg_node.base_model))

    # 3. Training Process
    faiss_maml_trainer = FaissMAMLTrainer(cfg_node, base_vl_model, bert_tokenizer, nlp_stanza, device)


    # 4: different training
    if cfg_node.exp_type in ["faiss_dgmaml", "faiss_maml"]:
        faiss_maml_trainer.faiss_mamal_training()



def main(args):

    # ---------------------------------
    # init cfg_node using yaml fuke
    # ---------------------------------
    cfg_node = CfgNode(new_allowed=True)
    cfg_node.merge_from_file(args.config_file)

    # --------------------------
    # modify cfg_node using args
    # --------------------------
    # normal part
    cfg_node.base_model = args.base_model
    cfg_node.dataset = args.dataset
    cfg_node.seed = args.seed
    cfg_node.output_dir = args.out_dir
    cfg_node.is_novel_comps = args.novel_comps
    cfg_node.exp_type = args.exp_type
    cfg_node.task_type = args.task_type
    cfg_node.val.val_item_num = cfg_node.val.batch_size * 100
    cfg_node.MetaTrain.fomaml = True if cfg_node.exp_type == "fomaml" else False        # str --> bool

    # meta part
    cfg_node.MetaTrain.inner_update_steps = args.inner_update_steps
    cfg_node.MetaTrain.episode_batch_size = args.episode_batch_size
    cfg_node.MetaTrain.episode_num = args.episode_num
    cfg_node.MetaTrain.target_comp_cpt_num = args.target_cpt_num
    cfg_node.MetaTrain.shot_num = args.shot_num
    cfg_node.MetaTrain.mask_sup_pair = str2bool(args.mask_sup_pair)
    cfg_node.MetaTrain.sup_weight = args.sup_weight

    # modify exp type
    cfg_node.MetaTrain.train_from_scratch = str2bool(args.train_from_scratch)
    if cfg_node.MetaTrain.train_from_scratch:
        cfg_node.exp_type += "FromScratch"


    # ----------------------
    # adding specific dirs
    # ----------------------
    # base_output_dir: loading the pretrained model
    add_base_model_dir_into_cfgnode(cfg_node)
    # meta_output_dir: recording the training process
    add_meta_exp_dir_into_cfgnode(cfg_node)
    # episode data dir
    add_episode_data_dir_into_cfgnode(cfg_node)

    # print config
    logger.info("Running with config:\n{}".format(cfg_node))

    # -----------------------------
    # save config to output_dir
    # -----------------------------
    cfg_file = os.path.join(cfg_node.meta_output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(cfg_file))
    save_cfg_node(cfg_node, cfg_file)

    # --------------------------------------------------------
    # init sweep config: not using and using yaml instead
    # --------------------------------------------------------
    """
    # 1. metric dict
    dictMetric = {
        'name': 'loss',
        'goal': 'minimize'
    }
    # 2. parameter
    dictParameters = {
        "MetaTrain.inner_update_steps":{
            'values': [1,2,3,4]
        },
        #"MetaTrain.target_comp_cpt_num":{
        #    'values': [1,2,3]
        #},
        #"MetaTrain.shot_num":{
        #    'values': [4,8,16,32]
        #}
    }
    # 3. sweep config
    dict_SweepCfg = {
        'method': 'grid',
    }
    # 4. add par and metric to sweep
    dict_SweepCfg['parameters'] = dictParameters
    dict_SweepCfg['metric'] = dictMetric
    """

    # ----------------------------------------------------------------
    # wandb_conifg:
    #   1. wandb flatten the names using dots in our backend
    #   2. dict access by dict_CfgNode['a']['b'] not dict_CfgNode.a.b
    #   3. avoid using dots in your config variable names, and use a dash or underscore instead.
    #   4. accesses wandb.config keys below the root, use [ ] syntax instead of . syntax
    # ----------------------------------------------------------------
    dict_CfgNode = convert_cfgnode_to_dict(cfg_node)

    # ---------------------------------------
    # sweep name using meta_output_dir:
    #   1. other wise random name
    #   2. '/tank/space/xugy07/MetaVLScratch/runs/mscoco/fomaml/vlbert_OuterLr_0.0001_InnerLr_0.0001_CompCptNum_1_SupNum_8_InnerStep_1_Epoch_8_Pretrain_True'
    # example name: mscoco_fomaml_vlbert_OuterLr_0.0001_InnerLr_0.0001_CompCptNum_1_SupNum_8_InnerStep_1_Epoch_8_Pretrain_True
    # ---------------------------------------
    sweep_name = "_".join(cfg_node.meta_output_dir.split("/")[-3:])
    sweep_name = sweep_name + "_" + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with wandb.init(project="MetaCompLearn",
                    entity="xugy07",
                    config=dict_CfgNode,
                    name=sweep_name,
                    mode='disabled'):     # otherwise just random name

        # Access all hyperparameter values through wandb.config
        dict_CfgNode = dict(wandb.config)
        print('-' * 20, "wandb_config", '-' * 20)
        print(json.dumps(dict_CfgNode, sort_keys=True, indent=4))
        print('-' * 50)

        # check whether we get sweep config
        cfg_node = CfgNode(init_dict = dict_CfgNode)
        print('-' * 20, "cfg_node", '-' * 20)
        print(cfg_node)
        print('-' * 50)

        # check whether they are the
        # do we need to re-assign the data
        meta_train(cfg_node)
        # train(config)



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

    # previous settings
    parser.add_argument('--base_model', type=str, default='vlbert')
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument('--config_file', type=str, default='{}/Config/maml_cfg.yaml'.format(ProjDir))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg', nargs='*')
    # parser.add_argument('--owutput_dir', type=str, default=out_dir)
    parser.add_argument('--out_dir', type=str, default=ExpDir)
    parser.add_argument('--novel_comps', action='store_true')
    parser.add_argument('--exp_type', choices=["ground", "supervise", "maml", "fomaml", "reptile", "dgmaml",
                                               "faiss_maml", "faiss_dgmaml"],
                        default="dgmaml")
    parser.add_argument('--task_type', choices=["random", "comp", "object", "faiss"],
                        default="faiss")

    # expose new settings for wandb
    parser.add_argument('--inner_update_steps', type=int, default=1)
    parser.add_argument('--episode_batch_size', type=int, default=8)
    parser.add_argument('--episode_num', type=int, default=2000)
    parser.add_argument('--target_cpt_num', type=int, default=1)
    parser.add_argument('--shot_num', type=int, default=8)
    parser.add_argument('--mask_sup_pair', type=str, default="True")
    parser.add_argument('--sup_weight', type=float, default=1.0)

    # ablaiton
    parser.add_argument('--train_from_scratch', type=str, default="False")

    args = parser.parse_args()

    # -----------------------
    # fix seed
    # -----------------------
    fix_seed(args.seed)

    # -----------------------
    # main func
    # -----------------------
    main(args)
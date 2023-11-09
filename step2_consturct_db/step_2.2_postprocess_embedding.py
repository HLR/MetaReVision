"""
post processing:

Dir:
/egr/research-hlr/xugy07/MetaCompLearnData/rawdata/mscoco/DataStore
1. organized by dataset name
2. seperate dir within each dir：
    2.1 DataStore
    2.2 CleanDataSplit
    2.3 img_feat

we have two files:
1. obj_datastore.pt
2. state_datastore.pt
"""

# load dict
import os
import json
import pickle
import torch
from ProjUtils.Constant import MSCOCOAllPair2ItemListJSON,\
                               MSCOCOPairDir, \
                               MSCOCOTrainSeenItemListPKL, \
                               MSCOCOObjDataStore, MSCOCOStateDataStore, MSCOCODataStore

# --------------------------------------
# load datastore value list: 600260
#   'raw_sent' = {str} 'A chef is preparing and decorating many small pastries.'
#   'img_id' = {int} 384029
#   'state_position' = {int} 6
#   'state_lemma' = {str} 'many'
#   'state_txt' = {str} 'many'
#   'obj_position' = {int} 8
#   'obj_lemma' = {str} 'pastry'
#   'obj_txt' = {str} 'pastries'
#   'split' = {str} 'train2014'
#   'pair_lemma' = {str} 'many_pastry'
#   'pair_txt' = {str} 'many_pastries'
#   'pair_root' = {str} 'many_pastry'
#   'state_type' = {str} 'adj'
# --------------------------------------
dict_TrainPair2ItemList = json.load(open(os.path.join(MSCOCOPairDir, MSCOCOAllPair2ItemListJSON.format("train2014")), "r"))
item_list = pickle.load(open(MSCOCOTrainSeenItemListPKL, "rb"))

# load datastore key list: 600260 * 384
obj_datastore = torch.load(MSCOCOObjDataStore)
state_datastore = torch.load(MSCOCOStateDataStore)
datastore = torch.cat([obj_datastore, state_datastore])

# save all datastore
torch.save(datastore, MSCOCODataStore)
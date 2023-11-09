"""
we have 3 levels of information:
1. sup_set and qry_set should put into a dataset
2. then put sup_set and qry_set into EpisodeItem
3. construct episode_list
4. then put episode_list into dataloader


-------------------------------------


refer: metawsd:
1. transformer:
2. allennlp: 0.9.0, very old version
"""
import json
from torch.utils.data import Dataset
from DataSet.NormalDataSet import FlickrNormalDataSet, MSCOCONormalDataSet

class FlickrOneEpisodeItemListDataSet(FlickrNormalDataSet):
    """
    one epiosde = one json file
    one json file = list of dictItems

    --------------------------------------

    this class is used to model "list of dictItemsOneEpoch"
    """

    def __init__(self, json_file, cfg_node, bert_tokenizer, nlp_stanza,
                 train_item_list = None, dict_TrainH5pyID2ImgID = None,
                 dict_TrainImgID2H5pyID = None, train_h5py_bbox_feat = None,
                 dict_TrainPair2ItemList = None):
        """
        :param cfg_node: information about list split
        :param json_file: list of dictItem
        """
        # init parent class
        super(FlickrOneEpisodeItemListDataSet, self).__init__(cfg_node,
                                                              "train",
                                                              bert_tokenizer,
                                                              nlp_stanza,
                                                              train_item_list,
                                                              dict_TrainH5pyID2ImgID,
                                                              dict_TrainImgID2H5pyID,
                                                              train_h5py_bbox_feat,
                                                              dict_TrainPair2ItemList)

        # re-write item_list
        with open(json_file, 'r', encoding='utf8') as f:
            self.item_list = json.load(f)

    def __len__(self):
        return super(FlickrOneEpisodeItemListDataSet, self).__len__()

    def __getitem__(self, idx):
        return super(FlickrOneEpisodeItemListDataSet, self).__getitem__(idx)



class MSCOCOOneEpisodeItemListDataSet(MSCOCONormalDataSet):
    """
    one epiosde = one json file
    one json file = list of dictItems

    --------------------------------------

    this class is used to model "list of dictItemsOneEpoch"
    """
    def __init__(self, json_file, cfg_node, bert_tokenizer, nlp_stanza, dict_TrainPair2ItemList, dict_ValPair2ItemList):
        """
        :param cfg_node: information about list split
        :param json_file: list of dictItem
        """
        # init parent class
        super(MSCOCOOneEpisodeItemListDataSet, self).__init__(cfg_node, "train", bert_tokenizer, nlp_stanza, dict_TrainPair2ItemList, dict_ValPair2ItemList)

        # re-write item_list
        with open(json_file, 'r', encoding='utf8') as f:
            self.item_list = json.load(f)

    def __len__(self):
        return super(MSCOCOOneEpisodeItemListDataSet, self).__len__()

    def __getitem__(self, idx):
        return super(MSCOCOOneEpisodeItemListDataSet, self).__getitem__(idx)



class OneEpisode:
    """ EpisodeItem it has own class """
    def __init__(self, sup_loader, qry_loader, task_name):
        self.sup_loader = sup_loader
        self.qry_loader = qry_loader
        self.base_task = task_name


class ListEpisodeDataset(Dataset):

    def __init__(self, episode_list):
        self.episode_list = episode_list

    def __len__(self):
        return len(self.episode_list)

    def __getitem__(self, index):
        return self.episode_list[index]
"""
this file is used to load information:
1. not used for training
2. but for task construciton

refer:
1. metaWSD: SemCorWSDDataset
2. Step5_ConstructTrainValTestList
"""

import os
import json, pickle
import tqdm
# MSCOCO constant
from ProjUtils.Constant import MSCOCOPairDir,\
    MSCOCOAllPair2ItemListJSON, MSCOCONovelPair2SynonPairDictPKL, \
    MSCOCONovelImgSetPKL, MSCOCOEpisodeTrainSeenCpt2ItemListPKL, \
    MSCOCO_H5PYID2IMGID_TRAIN
# Flickr constant
from ProjUtils.Constant import FlickrPair2ItemDir, \
    FlickrAllPair2ItemList, FlickrNovelPair2SynonPairPKL, \
    FlickrNovelImgSetPKL, FlickrEpisodeTrainSeenCpt2ItemList


def inv_dict(dic):
    return {v: k for k, v in dic.items()}

# train set
def load_train_imgs_with_bbox():
    dict_TrainH5pyID2ImgID = json.load(open(MSCOCO_H5PYID2IMGID_TRAIN))
    dict_TrainImgID2H5pyID = inv_dict(dict_TrainH5pyID2ImgID)
    train_img_list_having_bbox = dict_TrainImgID2H5pyID.keys()
    return train_img_list_having_bbox

def construct_dict_TrainSeenCpt2ItemList(dict_TrainPair2ItemList, noval_img_set, pkl_file, train_img_list_having_bbox):
    """
    original:
    1. dict_TrainPair2ItemList
    2. pair could be novel pair
    """
    # filter by novel_img_set
    dict_TrainSeenPair2ItemList_ByImgID = dict()
    dict_TrainSeenAttr2ItemList_ByImgID = dict()
    dict_TrainSeenObj2ItemList_ByImgID = dict()
    for pair, item_list in tqdm.tqdm(dict_TrainPair2ItemList.items()):
        for dictItem in item_list:
            img_id = dictItem['img_id']
            if img_id not in noval_img_set and img_id in train_img_list_having_bbox:

                # split from pair
                state, obj = dictItem['pair_txt'].split("_")

                # attr
                if state not in dict_TrainSeenAttr2ItemList_ByImgID:
                    dict_TrainSeenAttr2ItemList_ByImgID[state] = [dictItem]
                else:
                    dict_TrainSeenAttr2ItemList_ByImgID[state].append(dictItem)

                # obj
                if obj not in dict_TrainSeenObj2ItemList_ByImgID:
                    dict_TrainSeenObj2ItemList_ByImgID[obj] = [dictItem]
                else:
                    dict_TrainSeenObj2ItemList_ByImgID[obj].append(dictItem)

                if pair not in dict_TrainSeenPair2ItemList_ByImgID:
                    dict_TrainSeenPair2ItemList_ByImgID[pair] = [dictItem]
                else:
                    dict_TrainSeenPair2ItemList_ByImgID[pair].append(dictItem)
    # seen_item_num_by_imgID = sum([len(item_list) for pair, item_list in dict_TrainSeenPair2ItemList_ByImgID.items()])


    # filter by novel_pair
    """
    dict_TrainSeenPair2ItemList_ByPair = dict()
    novel_pair_list = dict_NovelSynonPair2RootPair.keys()
    for pair, item_list in dict_TrainPair2ItemList.items():
        if pair not in novel_pair_list:
            dict_TrainSeenPair2ItemList_ByPair[pair] = item_list
    seen_item_num_by_pair = sum([len(item_list) for pair, item_list in dict_TrainSeenPair2ItemList_ByPair.items()])
    """

    # ------------------------
    # check consistent
    # 1. filter by pair has more items
    # 2. one image can have novel pair caption, and no novel pair caption
    # ------------------------
    # print()
    # is_equal = (dict_TrainSeenPair2ItemList_ByPair == dict_TrainSeenPair2ItemList_ByImgID)

    dict_infor = {"dict_TrainSeenPair2ItemList" : dict_TrainSeenPair2ItemList_ByImgID,
                  "dict_TrainSeenAttr2ItemList" : dict_TrainSeenAttr2ItemList_ByImgID,
                  "dict_TrainSeenObj2ItemList" : dict_TrainSeenObj2ItemList_ByImgID}

    pickle.dump(dict_infor, open(pkl_file, "wb"))


class CompCptMetaInfor():
    """
    save all the meta information
    1.
    """
    def __init__(self, dataset):
        if dataset.lower() == "mscoco":
            # -----------------------------
            # MSCOCO:
            # 1. train --> train_seen and train_novel
            # 2. val --> val_seen and val_novel
            # -----------------------------
            # train_set: train_seen[:-500]
            # val_set: train_seen[-500:]
            # test_seen: val_seen
            # test_novel: val_novel + seen_novel
            # -----------------------------

            if not os.path.exists(MSCOCOEpisodeTrainSeenCpt2ItemListPKL):
                # pair mapping
                dict_PairMapping = pickle.load(open(MSCOCONovelPair2SynonPairDictPKL, "rb"))
                self.dict_TrainSynonPair2RootPair = dict_PairMapping['dict_TrainNovelSynonPair2RootPair']
                self.dict_TrainRootPair2SynonPair = dict_PairMapping['dict_TrainNovelRootPair2SynonPair']
                self.dict_ValSynonPair2RootPair = dict_PairMapping['dict_ValNovelSynonPair2RootPair']
                self.dict_ValRootPair2SynonPair = dict_PairMapping['dict_ValNovelRootPair2SynonPair']
                self.dict_NovelSynonPair2RootPair = dict_PairMapping['dict_NovelSynonPair2RootPair']

                # novel img set
                novel_img_set = pickle.load(open(MSCOCONovelImgSetPKL, "rb"))

                # item list
                # train_seen_item_list = pickle.load(open(MSCOCOTrainSeenItemList, "rb"))
                # val_seen_item_list = pickle.load(open(MSCOCOValSeenItemList, "rb"))

                # load train imgs with bbox
                train_img_list_having_bbox = load_train_imgs_with_bbox()

                # dict_TrainPair2ItemList
                train_file = os.path.join(MSCOCOPairDir, MSCOCOAllPair2ItemListJSON.format("train2014"))
                with open(train_file, "r") as json_file:
                    dict_TrainPair2ItemList = json.load(json_file)

                # filter train pair list
                construct_dict_TrainSeenCpt2ItemList(dict_TrainPair2ItemList,
                                                     novel_img_set,
                                                     MSCOCOEpisodeTrainSeenCpt2ItemListPKL,
                                                     train_img_list_having_bbox)

            dict_Infor = pickle.load(open(MSCOCOEpisodeTrainSeenCpt2ItemListPKL, "rb"))
            self.dict_TrainSeenPair2Itemlist = dict_Infor['dict_TrainSeenPair2ItemList']
            self.dict_TrainSeenAttr2Itemlist = dict_Infor['dict_TrainSeenAttr2ItemList']
            self.dict_TrainSeenObj2Itemlist = dict_Infor['dict_TrainSeenObj2ItemList']

        elif dataset.lower() == "flickr":
            if not os.path.exists(FlickrEpisodeTrainSeenCpt2ItemList):
                # pair mapping
                dict_PairMapping = pickle.load(open(FlickrNovelPair2SynonPairPKL, "rb"))
                self.dict_TrainSynonPair2RootPair = dict_PairMapping['dict_TrainSynonPair2RootPair']
                self.dict_TrainRootPair2SynonPair = dict_PairMapping['dict_TrainRootPair2SynonPair']
                self.dict_ValSynonPair2RootPair = dict_PairMapping['dict_ValSynonPair2RootPair']
                self.dict_ValRootPair2SynonPair = dict_PairMapping['dict_ValRootPair2SynonPair']
                self.dict_TestRootPair2SynonPair = dict_PairMapping['dict_TestRootPair2SynonPair']
                self.dict_TestRootPair2SynonPair = dict_PairMapping['dict_TestRootPair2SynonPair']
                self.dict_NovelSynonPair2RootPair = dict_PairMapping['dict_NovelSynonPair2RootPair']

                # novel img set
                novel_img_set = pickle.load(open(FlickrNovelImgSetPKL, "rb"))

                # item list
                # train_seen_item_list = pickle.load(open(MSCOCOTrainSeenItemList, "rb"))
                # val_seen_item_list = pickle.load(open(MSCOCOValSeenItemList, "rb"))

                # dict_TrainPair2ItemList
                train_file = os.path.join(FlickrPair2ItemDir, FlickrAllPair2ItemList.format("Train"))
                with open(train_file, "r") as json_file:
                    dict_TrainPair2ItemList = json.load(json_file)

                # filter train pair list
                construct_dict_TrainSeenCpt2ItemList(dict_TrainPair2ItemList, novel_img_set,
                                                     FlickrEpisodeTrainSeenCpt2ItemList)

            dict_Infor = pickle.load(open(FlickrEpisodeTrainSeenCpt2ItemList, "rb"))
            self.dict_TrainSeenPair2Itemlist = dict_Infor['dict_TrainSeenPair2ItemList']
            self.dict_TrainSeenAttr2Itemlist = dict_Infor['dict_TrainSeenAttr2ItemList']
            self.dict_TrainSeenObj2Itemlist = dict_Infor['dict_TrainSeenObj2ItemList']


        else:
            raise ValueError('Not Support Dataset In CompCptInfor!')



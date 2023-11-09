"""

Copy from:
GroundDataSet

---------------------------------------

Refer:
pretrain/data/datasets/coco_captions.py

"""
import os
import tqdm
import pickle, json, h5py
import logging
import torch
import random
from torch.utils.data import Dataset, DataLoader
from ProjUtils.Constant import FlickrAllPair2ItemList, FlickrPair2ItemDir, FlickrNovelImgSetPKL
from ProjUtils.Constant import FlickrTrainSeenItemList, FlickrValSeenItemList, \
    FlickrTestSeenItemList, FlickrTestNovelItemList, FlickrValNovelItemList, FlickrValItemList

# --------------------------------
# local constants
# --------------------------------
FEAT_BASE = '/tank/space/xugy07/ContMulModal/rawdata/flickr/img_feat'
TRAIN_FEAT = '{}/faster_rcnn_flickr_train.h5'.format(FEAT_BASE)
VAL_FEAT = '{}/faster_rcnn_flickr_val.h5'.format(FEAT_BASE)
TEST_FEAT = '{}/faster_rcnn_flickr_test.h5'.format(FEAT_BASE)

MAP_H5PYID2IMGID_TRAIN = '/tank/space/xugy07/ContMulModal/rawdata/flickr/flickr_train_map.json'
MAP_H5PYID2IMGID_VAL = '/tank/space/xugy07/ContMulModal/rawdata/flickr/flickr_val_map.json'
MAP_H5PYID2IMGID_TEST = '/tank/space/xugy07/ContMulModal/rawdata/flickr/flickr_test_map.json'

# --------------------------------
# help funcs
# --------------------------------
def inv_dict(dic):
    return {v: k for k, v in dic.items()}


class FlickrNormalDataSet(Dataset):

    def __init__(self, cfg_node, data_split, bert_tokenizer, nlp_stanza,
                 train_item_list = None, dict_TrainH5pyID2ImgID = None,
                 dict_TrainImgID2H5pyID = None, train_h5py_bbox_feat = None,
                 dict_TrainPair2ItemList = None):
        """
        Basically, we only need the data_split.
        bert_tokenizer and data_split for saving memory
        """
        # 1. init member variables
        self.data_split = data_split
        self.val_item_num = cfg_node.val.val_item_num

        # 2. load common infor
        if data_split.lower().startswith("val"):
            tmp_data_split = "Val"
        elif data_split.lower().startswith("train"):
            tmp_data_split = "Train"
        elif data_split.lower().startswith("test"):
            tmp_data_split = "Test"

        # meta-training
        if tmp_data_split == "Train" and dict_TrainPair2ItemList != None:
            self.dict_Pair2ItemList = dict_TrainPair2ItemList
        else:
            self.dict_Pair2ItemList = json.load(open(os.path.join(FlickrPair2ItemDir, FlickrAllPair2ItemList.format(tmp_data_split)), "r"))
        self.novel_img_set = pickle.load(open(FlickrNovelImgSetPKL, "rb"))

        # 3. load split-specific data-items
        if data_split.lower() == "train":
            if train_item_list == None:
                self.item_list = pickle.load(open(FlickrTrainSeenItemList, "rb"))
                self.dict_H5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_TRAIN))
                self.dict_ImgID2H5pyID = inv_dict(self.dict_H5pyID2ImgID)
                self.h5py_bbox_feat = h5py.File(TRAIN_FEAT, 'r')
            else:
                self.item_list = train_item_list
                self.dict_H5pyID2ImgID = dict_TrainH5pyID2ImgID
                self.dict_ImgID2H5pyID = dict_TrainImgID2H5pyID
                self.h5py_bbox_feat = train_h5py_bbox_feat
        elif data_split.lower() == "val_seen":
            self.item_list = pickle.load(open(FlickrValSeenItemList, "rb"))
            random.shuffle(self.item_list)
            self.item_list = self.item_list[:self.val_item_num]      # random truncate select val_seen set
            self.dict_H5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_VAL))
            self.dict_ImgID2H5pyID = inv_dict(self.dict_H5pyID2ImgID)
            self.h5py_bbox_feat = h5py.File(VAL_FEAT, 'r')
        elif data_split.lower() == "test_seen":
            self.item_list = pickle.load(open(FlickrTestSeenItemList, "rb"))
            self.dict_H5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_TEST))
            self.dict_ImgID2H5pyID = inv_dict(self.dict_H5pyID2ImgID)
            self.h5py_bbox_feat = h5py.File(TEST_FEAT, 'r')
        elif data_split.lower() in ["test_novel", "val_novel", "val"]:
            if data_split.lower() == "test_novel":
                self.item_list = pickle.load(open(FlickrTestNovelItemList, "rb"))
            elif data_split.lower() == "val_novel":
                self.item_list = pickle.load(open(FlickrValNovelItemList, "rb"))
            elif data_split.lower() == "val":
                self.item_list = pickle.load(open(FlickrValItemList, "rb"))
            # self.item_list = pickle.load(open(FlickrTestNovelItemList, "rb"))
            # load test
            self.dict_TestH5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_TEST))
            self.dict_TestImgID2H5pyID = inv_dict(self.dict_TestH5pyID2ImgID)
            self.test_h5py_bbox_feat = h5py.File(TEST_FEAT, 'r')
            # load train
            self.dict_TrainH5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_TRAIN))
            self.dict_TrainImgID2H5pyID = inv_dict(self.dict_TrainH5pyID2ImgID)
            self.train_h5py_bbox_feat = h5py.File(TRAIN_FEAT, 'r')
            # load val
            self.dict_ValH5pyID2ImgID = json.load(open(MAP_H5PYID2IMGID_VAL))
            self.dict_ValImgID2H5pyID = inv_dict(self.dict_ValH5pyID2ImgID)
            self.val_h5py_bbox_feat = h5py.File(VAL_FEAT, 'r')
        else:
            print("Data split not supported!")
            raise

        # 4: tools to construct
        self.bert_tokenizer = bert_tokenizer
        self.nlp_stanza = nlp_stanza



    def prepare_txt_tokens(self, dictItem):
        """
        you can refer:
        https://github.com/jackroos/VL-BERT/coco_captions.py
        1. basic_tokenizer
        2. random_word_wwm:
            2.1 wordpiece_tokenizer
        """
        # load infor
        raw_sent = dictItem["sent"]
        pair = dictItem["pair"]
        obj_pos = [dictItem["obj_position"]]
        state_pos = [dictItem["state_position"]]
        pair_pos = state_pos + obj_pos

        mask_attr = dictItem.get("mask_attr", True)
        mask_obj = dictItem.get("mask_obj", True)

        # -----------------------
        # basic tokenize
        # -----------------------
        bert_base_token_str_list = self.bert_tokenizer.basic_tokenizer.tokenize(raw_sent)
        # stata_token_str_list = self.staza(raw_sent).sentences[0].tokens

        # -----------------------
        # wordpiece tokenizer
        # -----------------------
        # 1. result
        wordpiece_output_tokens = []
        wordpiece_output_label = []
        re_state_pos, re_obj_pos = [], []
        # 2. iterate base tokens
        for token_idx, token in enumerate(bert_base_token_str_list):
            # sub_token each whole token
            sub_tokens = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)

            # mask attr
            if token_idx in state_pos and mask_attr:
                # add mask
                for sub_token in sub_tokens:
                    wordpiece_output_tokens.append("[MASK]")
                    re_state_pos.append(len(wordpiece_output_tokens) - 1)
                # add label
                for sub_token in sub_tokens:
                    try:
                        wordpiece_output_label.append(self.bert_tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        wordpiece_output_label.append(self.bert_tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))

            elif token_idx in obj_pos and mask_obj:
                # mask pair tokens
                for sub_token in sub_tokens:
                    wordpiece_output_tokens.append("[MASK]")
                    re_obj_pos.append(len(wordpiece_output_tokens) - 1)

                # add pair labels
                for sub_token in sub_tokens:
                    try:
                        wordpiece_output_label.append(self.bert_tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        wordpiece_output_label.append(self.bert_tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))

            else:
                # ------------------------------------------------------------
                # no masking token (will be ignored by loss function later)
                # ------------------------------------------------------------
                for sub_token in sub_tokens:
                    wordpiece_output_tokens.append(sub_token)
                    wordpiece_output_label.append(-1)

        return wordpiece_output_tokens, wordpiece_output_label, re_state_pos, re_obj_pos



    def prepare_visual_tokens(self, img_id):
        if self.data_split.lower() in ["test_novel", "val_novel", "val"]:
            if img_id in self.dict_TrainImgID2H5pyID:
                h5py_id = int(self.dict_TrainImgID2H5pyID[img_id])
                bbox_feats = torch.from_numpy(self.train_h5py_bbox_feat['bbox_features'][h5py_id]).float()
                bboxes = torch.from_numpy(self.train_h5py_bbox_feat['bboxes'][h5py_id]).float()
                bbox_num = torch.from_numpy(self.train_h5py_bbox_feat['num_boxes'][h5py_id]).long()
            if img_id in self.dict_TestImgID2H5pyID:
                h5py_id = int(self.dict_TestImgID2H5pyID[img_id])
                bbox_feats = torch.from_numpy(self.test_h5py_bbox_feat['bbox_features'][h5py_id]).float()
                bboxes = torch.from_numpy(self.test_h5py_bbox_feat['bboxes'][h5py_id]).float()
                bbox_num = torch.from_numpy(self.test_h5py_bbox_feat['num_boxes'][h5py_id]).long()
            if img_id in self.dict_ValImgID2H5pyID:
                h5py_id = int(self.dict_ValImgID2H5pyID[img_id])
                bbox_feats = torch.from_numpy(self.val_h5py_bbox_feat['bbox_features'][h5py_id]).float()
                bboxes = torch.from_numpy(self.val_h5py_bbox_feat['bboxes'][h5py_id]).float()
                bbox_num = torch.from_numpy(self.val_h5py_bbox_feat['num_boxes'][h5py_id]).long()
            dict_VisualTokens = {
                'bbox_feats': bbox_feats,       # [100(max bbox num), 1024],
                'bboxes': bboxes,  # [100, 4]
                'bbox_num': bbox_num,  # int
            }
        else:
            h5py_id = int(self.dict_ImgID2H5pyID[img_id])
            dict_VisualTokens = {
                'bbox_feats': torch.from_numpy(self.h5py_bbox_feat['bbox_features'][h5py_id]).float(),  # [100(max bbox num), 1024],
                'bboxes': torch.from_numpy(self.h5py_bbox_feat['bboxes'][h5py_id]).float(),  # [100, 4]
                'bbox_num': torch.from_numpy(self.h5py_bbox_feat['num_boxes'][h5py_id]).long(),  # int
            }
        return dict_VisualTokens


    def pad_or_truncate_token_list(self, txt_token_list, txt_label_list, length = 64):
        """
        default is 64:
        1. this is a different setting.2
        2. [txt_token_list] + [vis_token_list]
        it must have fixed length.
        """
        # determine length
        length = 64

        # trucate if long
        txt_token_list = txt_token_list[:length]
        txt_label_list = txt_label_list[:length]

        # pad if short
        return (txt_token_list + [self.bert_tokenizer.pad_token for _ in range(length - len(txt_token_list))]), \
               (txt_label_list + [-1 for _ in range(length - len(txt_label_list))]), \
               len(txt_token_list)


    def add_special_token(self, bert_token_str_list):
        """[cls] + str_list + [sep]"""
        return [self.bert_tokenizer.cls_token] + bert_token_str_list + [self.bert_tokenizer.sep_token]


    def __getitem__(self, item_idx):
        """
        [mask]: 103
        [cls]:  101
        [sep]:  102
        """
        # item
        dictItem = self.item_list[item_idx]

        # meta-infor
        img_id = dictItem['img_id']
        pair = dictItem['pair']
        attr, obj = pair.split("_")

        # state type
        state_type = dictItem["state_type"]

        # aligned bbox
        # align_bbox = torch.LongTensor([bbox_pos + 1 for bbox_pos in dictItem['aligned_bbox_list']])
        align_bbox = [bbox_pos + 1 for bbox_pos in dictItem['aligned_bbox_list']]

        # ------------------
        # Txt Tokens
        # 1. we use stanza to parse caption and extract pairs
        # 2. must be consistent with BertTokenizer
        # BertBaseTokenizer plays the role.
        # ------------------

        # step 1: deal with tokens
        bert_token_str_list, bert_token_label_list, re_state_pos, re_obj_pos = self.prepare_txt_tokens(dictItem)
        # step 2: add [cls] and [sep] and modify labels
        bert_token_str_list_with_speical = self.add_special_token(bert_token_str_list)
        bert_token_label_list_with_special = [-1] + bert_token_label_list + [-1]
        re_state_pos = [pos+1 for pos in re_state_pos]
        re_obj_pos = [pos + 1 for pos in re_obj_pos]
        # step 3: pad or truncate
        pad_bert_token_str_with_special, pad_bert_token_label_with_special, bert_token_len_with_special \
            = self.pad_or_truncate_token_list(bert_token_str_list_with_speical, bert_token_label_list_with_special)
        # step 4: token_str --> token_id
        pad_bert_token_id_with_special = self.bert_tokenizer.convert_tokens_to_ids(pad_bert_token_str_with_special)


        # 2.2 'input' (position 1) must be Tensor, not list: list --> tensor --> neg_one tensor
        # mask_txt_token_labels = -torch.ones_like(torch.LongTensor(pad_bert_token_id_with_special))   # all negative 1
        # 2.3 copy id
        # mask_txt_token_labels[1:1+len(bert_token_label_list)] = torch.LongTensor(bert_token_label_list)

        dict_TxtToken = {
            # meta-infor
            'img_id': img_id,
            'pair': pair,
            'attr': attr,
            'obj': obj,
            'state_pos': re_state_pos,
            'obj_pos': re_obj_pos,
            'state_type': state_type,
            # tensor-infor
            'mask_cap': torch.LongTensor(pad_bert_token_id_with_special).view(-1),
            'mask_label': torch.LongTensor(pad_bert_token_label_with_special).view(-1),
            'align_bbox': align_bbox
            # 'caption_len': text_len,
                # 'annotation_id': instance['instance_id']
            }

        # Visual Tokens
        img_id = dictItem["img_id"]
        dict_VisualToken = self.prepare_visual_tokens(img_id)

        dictReconsItem = {**dict_TxtToken, **dict_VisualToken}

        return dictReconsItem


    def __len__(self):
        return len(self.item_list)


if __name__ == "__main__":
    from DataSet.BatchCollator import BatchCollator
    dataset = FlickrNormalDataSet("Train")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=BatchCollator(data_set_name="Flickr"))

    for i, j in enumerate(tqdm.tqdm(dataloader)):
        print()
    # next(iter(dataloader))
    # feats, labels = next(iter(dataloader))

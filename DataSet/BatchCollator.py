"""
how to define the colltor
1. dictItem
2. extract items

-----------------------------------------------


"""
import torch

class BatchCollator:
    """
    base on different dataset, we have different field-names
    """
    TableFields = [
        'img_id',
        'pair',
        'attr',
        'obj',
        'state_pos',
        'obj_pos',
        'mask_cap',
        'mask_label',
        'align_bbox',
        'bbox_feats',
        'bboxes',
        'bbox_num',
    ]

    def __init__(self, data_set_name):
        """
        not need to inheritate any class
        """
        self.data_set_name = data_set_name.lower()
        self.cand_order_keys = self.TableFields



    def __call__(self, list_dictItems):
        """
        1. server for different VLModles
        2. should be shared with Flickr30K and MSCOCO
        """
        # define the results
        dict_BatchDictItem = {}


        # --------------------------------------------------------------
        # batching the batch input: list of dict --> dict of list
        # --------------------------------------------------------------
        # check each item
        for dictItem in list_dictItems:
            # check whether key existing in dictItem
            for field_key in self.cand_order_keys:
                if field_key in dictItem:
                    # put it into the results
                    if field_key not in dict_BatchDictItem:
                        dict_BatchDictItem[field_key] = []
                    dict_BatchDictItem[field_key].append(dictItem[field_key])


        # --------------------
        # list --> tensor
        # --------------------
        # case 1: list of tensor --> stacked tensor
        for field_key in dict_BatchDictItem:
            if torch.is_tensor(dict_BatchDictItem[field_key][0]):
                dict_BatchDictItem[field_key] = torch.stack(dict_BatchDictItem[field_key])
            elif type(dict_BatchDictItem[field_key][0]) is int:
                dict_BatchDictItem[field_key] = torch.LongTensor(dict_BatchDictItem[field_key])


        # ------------------------------------------
        # the result is still organized as dict
        # ------------------------------------------
        return dict_BatchDictItem, list_dictItems


def batch_collator_func(batch_DictItems):
    """
    two ways to implement out collator:
    1. one class
    2. one function
    """
    pass

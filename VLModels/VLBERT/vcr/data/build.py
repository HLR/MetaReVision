import torch.utils.data

from .datasets import *
from . import samplers
from .transforms.build import build_transforms
from .collate_batch import BatchCollator
import pprint

DATASET_CATALOGS = {'vcr': VCRDataset}


def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "DataSet not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size):
    if aspect_grouping:
        group_ids = dataset.group_ids
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False
        )
    return batch_sampler


def make_dataloader(cfg, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None,
                    expose_sampler=False):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        ann_file = cfg.dataset.TRAIN_ANNOTATION_FILE
        image_set = cfg.dataset.TRAIN_IMAGE_SET
        aspect_grouping = cfg.TRAIN.ASPECT_GROUPING
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        mask_vl_modeling = cfg.dataset.MASK_VL_MODELING if 'MASK_VL_MODELING' in cfg.dataset else False
        mask_language_modeling = cfg.NETWORK.BERT_WITH_MLM_LOSS if 'BERT_WITH_MLM_LOSS' in cfg.NETWORK else False
    elif mode == 'val':
        ann_file = cfg.dataset.VAL_ANNOTATION_FILE
        image_set = cfg.dataset.VAL_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        mask_vl_modeling = False
        mask_language_modeling = False
        if 'MASK_VL_MODELING' in cfg.dataset and cfg.dataset.MASK_VL_MODELING and cfg.NETWORK.FOR_MASK_VL_MODELING_PRETRAIN:
            mask_vl_modeling = True
    else:
        ann_file = cfg.dataset.TEST_ANNOTATION_FILE
        image_set = cfg.dataset.TEST_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        mask_vl_modeling = False
        mask_language_modeling = False

    transform = build_transforms(cfg, mode)

    if dataset is None:
        kwargs = {'mask_vl_modeling': mask_vl_modeling,
                  'mask_language_modeling': mask_language_modeling}
        if mask_vl_modeling:
            kwargs['mask_replace_only_same_cls'] = cfg.dataset.MASK_REPLACE_ONLY_SAME_CLS
            kwargs['mask_master_ind_random'] = (not cfg.NETWORK.FOR_MASK_VL_MODELING_PRETRAIN)
            kwargs['mask_vl_modeling_mask_prob'] = cfg.dataset.MASK_VL_MODELING_MASK_PROB
            kwargs['mask_vl_modeling_replace_prob'] = cfg.dataset.MASK_VL_MODELING_REPLACE_PROB
        try:
            kwargs['qa2r_noq'] = cfg.dataset.QA2R_NOQ
            kwargs['qa2r_aug'] = cfg.dataset.QA2R_AUG
        except AttributeError:
            pass
        try:
            kwargs['basic_align'] = cfg.dataset.BASIC_ALIGN
        except AttributeError:
            pass
        try:
            kwargs['with_lg'] = cfg.NETWORK.GNN.WITH_LG_LAYER
            kwargs['with_kg'] = cfg.NETWORK.GNN.WITH_KG
            kwargs['kg_path'] = cfg.dataset.__getattribute__('{}_KG_PATH'.format(mode.upper()))
            kwargs['kg_word_embed'] = cfg.dataset.__getattribute__('{}_KG_WORD_EMBED'.format(mode.upper()))
        except AttributeError:
            pass
        try:
            kwargs['kg_path'] = cfg.dataset.__getattribute__('{}_KG_PATH'.format(mode.upper()))
            kwargs['fact_path'] = cfg.dataset.__getattribute__('{}_KG_PATH'.format(mode.upper()))
        except AttributeError:
            pass
        try:
            kwargs['expression_file'] = cfg.dataset.__getattribute__('{}_EXPRESSION_FILE'.format(mode.upper()))
        except AttributeError:
            pass

        try:
            kwargs['kg_vocab_file'] = cfg.NETWORK.KB_NODE_VOCAB
        except AttributeError:
            pass

        try:
            kwargs['caption_file'] = cfg.dataset.__getattribute__('{}_CAPTION_FILE'.format(mode.upper()))
        except AttributeError:
            pass

        print('Dataset kwargs:')
        pprint.pprint(kwargs)

        dataset = build_dataset(dataset_name=cfg.dataset.dataset, ann_file=ann_file, image_set=image_set,
                                root_path=cfg.dataset.ROOT_PATH, data_path=cfg.dataset.DATASET_PATH,
                                pair_name=(mode == 'test'), task=cfg.dataset.TASK, transform=transform,
                                zip_mode=cfg.dataset.ZIP_MODE, cache_mode=cfg.dataset.CACHE_MODE,
                                ignore_db_cache=cfg.dataset.IGNORE_DB_CACHE,
                                only_use_relevant_dets=cfg.dataset.ONLY_USE_RELEVANT_DETS,
                                add_image_as_a_box=cfg.dataset.ADD_IMAGE_AS_A_BOX,
                                aspect_grouping=aspect_grouping,
                                mask_size=(cfg.dataset.MASK_SIZE, cfg.dataset.MASK_SIZE),
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME,
                                **kwargs)

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size)
    collator = BatchCollator(dataset=dataset, append_ind=cfg.dataset.APPEND_INDEX)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collator)
    if expose_sampler:
        return dataloader, sampler

    return dataloader

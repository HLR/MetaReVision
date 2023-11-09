import torch.utils.data

from .datasets import *
from . import samplers
from .transforms.build import build_transforms
from .collate_batch import BatchCollator

DATASET_CATALOGS = {'refcoco+': RefCOCO}


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
        boxes = cfg.dataset.TRAIN_BOXES
    elif mode == 'val':
        ann_file = cfg.dataset.VAL_ANNOTATION_FILE
        image_set = cfg.dataset.VAL_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        boxes = cfg.dataset.VAL_BOXES
    else:
        ann_file = cfg.dataset.TEST_ANNOTATION_FILE
        image_set = cfg.dataset.TEST_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        boxes = cfg.dataset.TEST_BOXES

    transform = build_transforms(cfg, mode)

    if dataset is None:

        dataset = build_dataset(dataset_name=cfg.dataset.dataset, ann_file=ann_file, image_set=image_set,
                                boxes=boxes, proposal_source=cfg.dataset.PROPOSAL_SOURCE,
                                answer_vocab_file=cfg.dataset.ANSWER_VOCAB_FILE,
                                root_path=cfg.dataset.ROOT_PATH, data_path=cfg.dataset.DATASET_PATH,
                                test_mode=(mode == 'test'), transform=transform,
                                zip_mode=cfg.dataset.ZIP_MODE, cache_mode=cfg.dataset.CACHE_MODE,
                                cache_db=True if (rank is None or rank == 0) else False,
                                ignore_db_cache=cfg.dataset.IGNORE_DB_CACHE,
                                add_image_as_a_box=cfg.dataset.ADD_IMAGE_AS_A_BOX,
                                aspect_grouping=aspect_grouping,
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME)

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

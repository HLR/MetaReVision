"""
here we define the interface for VLData
1. multiple val_dataset, seen and novel


2.

"""
from torch.utils.data import DataLoader
import pytorch_lightning as plt

from DataSet.NormalDataSet import FlickrNormalDataSet, MSCOCONormalDataSet

class VLDataPltInterface(plt.LightningDataModule):

    def __init__(self, cfg_node, bert_tokenizer,  nlp_stanza):
        super().__init__()

        # save all config
        self.cfg_node = cfg_node

        # extract some module specific
        self.dataset = cfg_node.dataset      # flickr or mscoco
        self.num_workers = cfg_node.num_workers
        self.train_batch_size = cfg_node.train.batch_size
        self.val_batch_size = cfg_node.val.batch_size

        # nlp tools
        self.bert_tokenizer = bert_tokenizer
        self.nlp_stanza = nlp_stanza



    def prepare_data(self) -> None:
        """
        run on one gpu and take nlp for example:
        1. download the data
        2. raw_txt --> tokenizer
        3. vocab
        4. save the vocab
        """
        pass


    def setup(self, stage):
        """
        here, we should:
        1. prepare the dataset
        2. then we can use these datasets to initlize the dataloders

        done on every process
        """
        if stage == 'fit': # train needs train and val
            if self.dataset.startswith('m'):
                # mscoco
                self.train_ds = MSCOCONormalDataSet(self.cfg_node, "Train", self.bert_tokenizer, self.nlp_stanza)
                self.val_seen_ds = MSCOCONormalDataSet(self.cfg_node, "Val_Seen", self.bert_tokenizer, self.nlp_stanza)  # capitalized to match file name
                self.val_novel_ds = MSCOCONormalDataSet(self.cfg_node, "Val_Novel", self.bert_tokenizer, self.nlp_stanza)
            elif self.dataset.startswith('f'):
                # flickr
                self.train_ds = FlickrNormalDataSet(self.cfg_node, "Train", self.bert_tokenizer, self.nlp_stanza)
                self.val_seen_ds = FlickrNormalDataSet(self.cfg_node, "Val_Seen", self.bert_tokenizer, self.nlp_stanza)  # capitalized to match file name
                self.val_novel_ds = FlickrNormalDataSet(self.cfg_node, "Val_Novel", self.bert_tokenizer, self.nlp_stanza)
        else:
            raise ValueError('unknown DataSet {}'.format(self.dataset))


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=cfg_node.train.batch_size, shuffle=True, collate_fn=batch_collator)


    def val_dataloader(self):
        pass


    def test_dataloader(self):
        pass



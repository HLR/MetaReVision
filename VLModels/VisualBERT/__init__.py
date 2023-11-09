"""

-----------------------------------------------------

VisualBert:

https://github.com/uclanlp/visualbert/blob/e49b61906dff12f1b2992226ae84c538fcf438e4/visualbert/pytorch_pretrained_bert/modeling.py#L1169

1. TrainVisualBERTObjective:
        1.1 forward(
                input_ids,
                token_type_ids,
                input_mask,

                visual_embeddings,
                position_embeddings_visual,
                image_mask,
            )
        1.2 self.bert = BertVisualModel(config)
2. BertVisualModel:
        2.1 embedding_output = self.embeddings
            2.1.1 words_embeddings = self.word_embeddings(input_ids)
            2.1.2
        2.3 embeddings = torch.cat((embeddings, v_embeddings), dim = 1)
### 3. it has the alignment, which kind of alignment it has?

---------------------------------------------------------------------------------

LXMERTTorchDataset:
1. organized by dictItem
2. one data item
        uid = datum['uid']
        img_id = datum['img_id']
        sent=datum['sent'].lower()
3. also return one item:
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label,
            use_visual_tag_flag=self.use_visual_tag_flag
        )
4. finally convert to one item:
        # unsupervised_visualbert/src/pretrain/lxmert_data.py
        self.convert_example_to_features(example, args.get("max_seq_length", 20), self.tokenizer)
        1. padding 20 txt_tokens
        2. other copy the max length
5. features = InputFeatures()
        InputFeatures is just a struct
        InputFeatures(object):
            init, nothing else
6. Read until the end of the file: hybrid_embedding
        if example.use_visual_tag_flag and example.visual_feats[0] is not None:  # Let's do a hybrid embedding



Conclusion: [txt_token, pad, vis_token]

------------------------------------------------------------

"""
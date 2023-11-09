"""
Class hierachy:

basemodel for init parameter
    |
    |
    VisualLinguisticBert for embedding
                    |
                    |
                    VisualLinguisticBertForPretraining



-------------------------------------------------------------------------------------------
Origianal VLBERT implmentation: why we need
1. two special tokens before text
2. change vlbert tokenizer


Model:
https://github.com/jackroos/VL-BERT/tree/4373674cbf2bcd6c09a2c26abfdb6705b870e3be/common

Dataset:
https://github.com/jackroos/VL-BERT/blob/4373674cbf2bcd6c09a2c26abfdb6705b870e3be/pretrain/data/datasets/coco_captions.py

otherwise really strange setting:
text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']


We don't add special tokens:
self.end_embedding = nn.Embedding(1, config.hidden_size)                                [END] defined in vlbert
self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)  [IMG] defined for specific head, like refcoco/modules/resnet_vlbert_for_refcoco.py


how to use [end]
# init tensor
vl_embeddings = text_vl_embeddings.new_zeros((bs, max_length, vl_embed_size))
# copy text
vl_embeddings[grid_pos < text_end] = text_vl_embeddings[text_mask]
# copy visual
vl_embeddings[(grid_pos >= text_end) & (grid_pos < object_end)]  = object_vl_embeddings[object_mask]
# end embedding, just emb values
vl_embeddings[grid_pos == object_end] = self.end_embedding(_zero_id)
-------------------------------------------------------------------------------------------



----------------------------------
how layer-norm works:
----------------------------------
1. for example, we have 1000 dim data
2. person == [100 dim description]
    2.1 sex, age ...
3. each dimension has its own meaning
4. each diemsnion has its mean and var
5. weight + bias
    5.1 data dimension
    5.2 not scalar
"""

import torch
import torch.nn as nn
from ..external.pytorch_pretrained_bert.modeling import BertLayerNorm, BertEncoder, BertPooler, ACT2FN, BertOnlyMLMHead

# todo: add this to config
NUM_SPECIAL_WORDS = 1000


class BaseModel(nn.Module):
    """ all the base models are used to initilize parameters """
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented



class VisualLinguisticBert(BaseModel):
    def __init__(self, config, language_pretrained_model_path=None):
        super(VisualLinguisticBert, self).__init__(config)

        self.config = config

        # ----------------------
        # embeddings
        # ----------------------
        # 30522 * 384
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # 1 * 384
        self.end_embedding = nn.Embedding(1, self.config.hidden_size)
        # 512 * 384: 512 is the same as Bert
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, config.hidden_size)
        # 3 * 384
        self.token_type_embeddings = nn.Embedding(self.config.type_vocab_size, config.hidden_size)

        # ----------------------------------------
        # norm layer implement by bert, simple
        # ----------------------------------------
        self.embedding_LayerNorm = BertLayerNorm(self.config.hidden_size, eps=1e-12)

        # -------------------
        # dropout with 0.1 rate
        # -------------------
        self.embedding_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # -----------------------------------
        # for compatibility of roberta: -1
        # -----------------------------------
        self.position_padding_idx = config.position_padding_idx

        # --------------------------------------------------------------------------------
        # visual transform because we working on the hidden space with dimension 384
        # --------------------------------------------------------------------------------
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(self.config.visual_size, self.config.hidden_size)
            self.visual_1x1_object = nn.Linear(self.config.visual_size, self.config.hidden_size)

        # ------------------
        # layer norm
        # ------------------
        if config.visual_ln:
            self.visual_layerNorm_for_text = BertLayerNorm(self.config.hidden_size, eps=1e-12)
            self.visual_ln_for_bbox = BertLayerNorm(self.config.hidden_size, eps=1e-12)
        else:
            visual_scale_text = nn.Parameter(torch.as_tensor(self.config.visual_scale_text_init, dtype=torch.float),
                                             requires_grad=True)
            self.register_parameter('visual_scale_text', visual_scale_text)
            visual_scale_object = nn.Parameter(torch.as_tensor(self.config.visual_scale_object_init, dtype=torch.float),
                                               requires_grad=True)
            self.register_parameter('visual_scale_object', visual_scale_object)

        # -----------------------
        # encoder: 6 BertLayers
        #
        # Each BertLayer has:
        # 1. BertAttention(config)
        #   1.1 SelfAttention: input --> hidden
        #   1.2 BertSelfOutput: res_connect, dense(hidden) --> dropout(hidden) --> layernorm(hidden + input)
        # 2. BertIntermediate(config)
        #   2.1 dense: 384 --> 1536
        #   2.2 act: gelu
        # 3. BertOutput(config)
        #   ResConnect: dense 1536 --> 384 --> dropout --> layernorm(hidden + input)
        #
        # attention_output = self.attention(hidden_states, attention_mask, output_attention_probs=output_attention_probs)
        # intermediate_output = self.intermediate(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)
        # -----------------------
        self.bert_encoder = BertEncoder(config)

        # ------------------------------------
        # pooler: no pooler currently
        # ------------------------------------
        if self.config.with_pooler:
            self.pooler = BertPooler(config)


        # -----------------------
        # init weights
        # -----------------------
        self.apply(self.init_weights)

        # ------------------------------------
        # fill with 0, that's not reasonable
        # ------------------------------------
        if config.visual_ln:
            self.visual_layerNorm_for_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_for_bbox.weight.data.fill_(self.config.visual_scale_object_init)

        # -----------------------
        # load language pretrained model
        # -----------------------
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

        # -----------------------
        # training the embedding
        # -----------------------
        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False
            self.special_word_embeddings = nn.Embedding(NUM_SPECIAL_WORDS, config.hidden_size)
            self.special_word_embeddings.weight.data.copy_(self.word_embeddings.weight.data[:NUM_SPECIAL_WORDS])


    def word_embeddings_wrapper(self, input_ids):
        """
        special work embedding
        """
        if self.config.word_embedding_frozen:
            word_embeddings = self.word_embeddings(input_ids)
            word_embeddings[input_ids < NUM_SPECIAL_WORDS] \
                = self.special_word_embeddings(input_ids[input_ids < NUM_SPECIAL_WORDS])
            return word_embeddings
        else:
            return self.word_embeddings(input_ids)


    def forward(self,
                txt_token_ids,
                txt_token_type,
                txt_visual_token,
                txt_token_mask,
                bbox_concat_vl_embeddings,
                bbox_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False,
                output_attention_probs=False):
        """
        all the base models are used to initilize parameters (nn.Module, not from bert)

        Txt part:
        1. token_id
        2. token_visual_feat, the whole img feat
        3. mask
        4. type: A B (0,1)

        Visual part:
        1. visual_concat_vl_emb
        2. mask

        vlbert follow Bert's setting, but lxnert is different:
        1. lxmert: [txt, pad, bbox(32)]
        2. vlbert: [txt(64), bbox, pad]
        """
        # get seamless concatenate embeddings and mask
        embedding_output, vl_attn_mask, new_text_mask, new_bbox_mask = self.embedding(txt_token_ids,
                                                                                        txt_token_type,
                                                                                        txt_visual_token,
                                                                                        txt_token_mask,
                                                                                        bbox_concat_vl_embeddings,
                                                                                        bbox_mask)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = vl_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_attention_mask = 1.0 - extended_attention_mask
        # extended_attention_mask[extended_attention_mask != 0] = float('-inf')


        # -----------------------------------------
        # embedding --> self-attn(bert_encoder)
        # -----------------------------------------
        if output_attention_probs:
            encoded_layers, attention_probs = self.bert_encoder(embedding_output,
                                                                extended_attention_mask,
                                                                output_all_encoded_layers=output_all_encoded_layers,
                                                                output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.bert_encoder(embedding_output,
                                               extended_attention_mask,
                                               output_all_encoded_layers=output_all_encoded_layers,
                                               output_attention_probs=output_attention_probs)

        # last layer output, last element of the list
        sequence_output = encoded_layers[-1]    # [8 31 384]

        # None: no pool layer, [cls] label
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None

        # -------------------------------------------------------------
        # choices:
        # 1. Bert has 12 layers, but we only define 7
        # 2. we can choose last layer output, or store all layer output
        # -------------------------------------------------------------
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # --------------------
        # output seperately
        # --------------------
        if output_text_and_object_separately:
            # wrapped using list
            if not output_all_encoded_layers:
                encoded_layers = [encoded_layers]

            # returned results
            encoded_layers_text = []
            encoded_layers_bbox = []

            for encoded_layer in encoded_layers:
                # ------------------
                # scan each layer
                # ------------------
                ori_max_text_len = txt_token_ids.shape[1]  # 64
                ori_max_bbox_len = bbox_concat_vl_embeddings.shape[1]  # 100
                encoded_layer_text = encoded_layer[:, :ori_max_text_len]
                encoded_layer_bbox = encoded_layer.new_zeros((encoded_layer.shape[0],
                                                              ori_max_bbox_len,
                                                              encoded_layer.shape[2]))  # 32 * 100 * 384
                encoded_layer_bbox[bbox_mask] = encoded_layer[new_bbox_mask]
                encoded_layers_text.append(encoded_layer_text)
                encoded_layers_bbox.append(encoded_layer_bbox)

            if not output_all_encoded_layers:
                # extract data from list
                encoded_layers_text = encoded_layers_text[0]        # list --> one element
                encoded_layers_bbox = encoded_layers_bbox[0]        # list --> one element

            if output_attention_probs:
                return encoded_layers_text, encoded_layers_bbox, pooled_output, attention_probs
            else:
                return encoded_layers_text, encoded_layers_bbox, pooled_output
        else:
            if output_attention_probs:
                return encoded_layers, pooled_output, attention_probs
            else:
                return encoded_layers, pooled_output


    def embedding(self,
                  txt_token_ids,
                  txt_token_type,
                  txt_visual_token,
                  txt_token_mask,
                  bbox_concat_vl_embeddings,
                  bbox_mask):
        """
        1. Emb layer has [end]
        2. it has many specific tokens
        """

        """---------------------------  text encoding  ---------------------------"""
        # -----------------------------------
        # text_linguistic_embedding
        # word_embedding: 30600 * 384
        # 32 * 64 --> 32 * 64 * 384
        # -----------------------------------
        text_linguistic_token = self.word_embeddings_wrapper(txt_token_ids)
        if self.visual_1x1_text is not None:
            txt_visual_token = self.visual_1x1_text(txt_visual_token)

        # ---------------------------------------
        # text_visual_embeddings: bertlayernorm
        # ---------------------------------------
        if self.config.visual_ln:
            txt_visual_token = self.visual_layerNorm_for_text(txt_visual_token)
        else:
            txt_visual_token *= self.visual_scale_text

        # ---------------------------------------------
        # adding two parts together: 32 * 64 * 384
        # ---------------------------------------------
        text_vl_token = text_linguistic_token + txt_visual_token



        """---------------------------  visual/bbox encoding  ---------------------------"""
        # -----------------------------------------------
        # bbox visual feats: visual_token + txt_token
        # -----------------------------------------------
        bbox_visual_embeddings = bbox_concat_vl_embeddings[:, :, :self.config.visual_size]
        if self.visual_1x1_object is not None:
            bbox_visual_embeddings = self.visual_1x1_object(bbox_visual_embeddings)


        if self.config.visual_ln:
            bbox_visual_embeddings = self.visual_ln_for_bbox(bbox_visual_embeddings)
        else:
            bbox_visual_embeddings *= self.visual_scale_object

        # -------------------------
        # bbox linguistic feats
        # -------------------------
        # MLM has prepared the bbox_vl_embeddings:
        bbox_linguistic_embeddings = bbox_concat_vl_embeddings[:, :, self.config.visual_size:]
        bbox_concat_vl_embeddings = bbox_linguistic_embeddings + bbox_visual_embeddings

        # ----------------
        # some statistics
        # ----------------
        batch_size = text_vl_token.size(0)
        vl_embed_dim = text_vl_token.size(-1)


        # ---------------------------------------------
        # orgainize the data based on current batch
        # 1. max vl length based on current batch
        # 2. organize based on the max length
        # 3. txt_token and vis_token are the same
        # ---------------------------------------------
        # tmp = text_mask.sum(1)  # sum will delete the dimension, with shape [32], like [1,2,3]
        # tmp2 = bbox_mask.sum(1) # like [4,5,1]
        # +1 because  we have the end embedding
        # ---------------------------------------------
        max_length = (txt_token_mask.sum(1) + bbox_mask.sum(1)).max() + 1

        # ----------------------
        # both: 32 * 49
        # grid_ind:
        #   [
        #        [0 0 0]
        #        [1 1 1]
        #         ...
        #   ]
        # grid_pos
        #   [
        #       [0,1,2]
        #       [0,1,2]
        #       ...
        #   ]
        # ----------------------
        grid_x, grid_y = torch.meshgrid(torch.arange(batch_size, dtype=torch.long, device=text_vl_token.device),
                                            torch.arange(max_length, dtype=torch.long, device=text_vl_token.device))

        text_end = txt_token_mask.sum(1, keepdim=True)
        bbox_end = text_end + bbox_mask.sum(1, keepdim=True)


        # ---------------------------------------------------------------------
        # vector_emb:
        # 1. seamlessly concatenate visual linguistic embeddings of text and bbox
        # 2. text then bbox
        # 3. init as zero
        # 4. then fill
        # ---------------------------------------------------------------------
        vl_embeddings = text_vl_token.new_zeros((batch_size, max_length, vl_embed_dim))
        vl_embeddings[grid_y < text_end] = text_vl_token[txt_token_mask]
        vl_embeddings[(grid_y >= text_end) & (grid_y < bbox_end)]  = bbox_concat_vl_embeddings[bbox_mask]

        # ----------------------------------
        # we really have end embedding
        # 1. end token has no visual or text
        # 2. just simple embedding
        # ----------------------------------
        zero_index_for_end_token = torch.zeros((batch_size,), dtype=torch.long, device=text_vl_token.device)
        vl_embeddings[grid_y == bbox_end] = self.end_embedding(zero_index_for_end_token)

        # -------------------------------------------------------
        # token type embeddings / segment embeddings
        # token_type_embeddings: 3 * 384, including A, B and C
        # -------------------------------------------------------
        token_type_ids = txt_token_type.new_zeros((batch_size, max_length))
        token_type_ids[grid_y < text_end] = txt_token_type[txt_token_mask]
        token_type_ids[(grid_y >= text_end) & (grid_y <= bbox_end)] = 2
        token_type_embeddings = self.token_type_embeddings(token_type_ids)


        """
        from the embedding, we can see that:
        1. txt_token and img_token are the same, just tokens
        2. for one item, position indexed by 0 1 2 3 ...
        3. for batch items, batch_max then the same embedding
        4. but we need mask
        5. then we can get all inforamtion 
        """
        # ---------------------------------------
        # position embeddings, importatnt: 32 * 49(current max length)
        # this is related to deep/shallow copy:
        # ---------------------------------------
        # position_ids = grid_pos   # shallow copy, share memory
        position_ids = grid_y.clone().detach()     # change to deep copy and it works
        if self.config.obj_pos_id_relative:
            position_ids[(grid_y >= text_end) & (grid_y < bbox_end)] \
                = text_end.expand((batch_size, max_length))[(grid_y >= text_end) & (grid_y < bbox_end)]
            position_ids[grid_y == bbox_end] = (text_end + 1).squeeze(1) + self.position_padding_idx + 1
        else:
            assert False, "Don't use position id 510/511 for objects and [END]!!!"
            position_ids[(grid_y >= text_end) & (grid_y < bbox_end)] = self.config.max_position_embeddings - 2
            position_ids[grid_y == bbox_end] = self.config.max_position_embeddings - 1
        position_embeddings = self.position_embeddings(position_ids)


        # -----------------
        # building mask
        # -----------------
        mask = txt_token_mask.new_zeros((batch_size, max_length))
        mask[grid_y <= bbox_end] = 1


        # ---------------------
        # final sum embedding
        # ---------------------
        embeddings = vl_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings, mask, grid_y < text_end, (grid_y >= text_end) & (grid_y < bbox_end)



    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        unexpected_keys = []
        for k, v in pretrained_state_dict.items():
            if k.startswith('bert.'):
                k = k[len('bert.'):]
            elif k.startswith('roberta.'):
                k = k[len('roberta.'):]
            else:
                unexpected_keys.append(k)
                continue
            if 'gamma' in k:
                k = k.replace('gamma', 'weight')
            if 'beta' in k:
                k = k.replace('beta', 'bias')
            if k.startswith('encoder.'):
                k_ = k[len('encoder.'):]
                if k_ in self.bert_encoder.state_dict():
                    encoder_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            elif k.startswith('embeddings.'):
                k_ = k[len('embeddings.'):]
                if k_ == 'word_embeddings.weight':
                    self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                            device=self.word_embeddings.weight.data.device)
                elif k_ == 'position_embeddings.weight':
                    self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                device=self.position_embeddings.weight.data.device)
                elif k_ == 'token_type_embeddings.weight':
                    self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                        dtype=self.token_type_embeddings.weight.data.dtype,
                        device=self.token_type_embeddings.weight.data.device)
                    if v.size(0) == 1:
                        # Todo: roberta token type embedding
                        self.token_type_embeddings.weight.data[1] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        self.token_type_embeddings.weight.data[2] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)

                elif k_.startswith('LayerNorm.'):
                    k__ = k_[len('LayerNorm.'):]
                    if k__ in self.embedding_LayerNorm.state_dict():
                        embedding_ln_pretrained_state_dict[k__] = v
                    else:
                        unexpected_keys.append(k)
                else:
                    unexpected_keys.append(k)
            elif self.config.with_pooler and k.startswith('pooler.'):
                k_ = k[len('pooler.'):]
                if k_ in self.pooler.state_dict():
                    pooler_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            else:
                unexpected_keys.append(k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.bert_encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)



class VisualLinguisticBertForPretraining(VisualLinguisticBert):
    """
    pretraining wrapper for VisualLinguisticBert
    """

    def __init__(self,
                 config,
                 language_pretrained_model_path=None,
                 with_rel_head=True,
                 with_mlm_head=True,
                 with_mvrc_head=True):
        """
        basic config + head types
        """

        super(VisualLinguisticBertForPretraining, self).__init__(config, language_pretrained_model_path=None)

        self.with_rel_head = with_rel_head
        self.with_mlm_head = with_mlm_head
        self.with_mvrc_head = with_mvrc_head

        # -------------------------------------------------
        # Bert model has two parts:
        # 1. embedding part: embedding for each token
        # 2. bert self-attn encoder
        # 3. head for updating tokens, we have three heads here:
        #   3.1 BertHead (from other code)
        #   3.2 model specific heads:
        #       3.2.1 rel_head
        #       3.2.2 mvrc_head
        # -------------------------------------------------
        if with_rel_head:
            self.relationsip_head = VisualLinguisticBertRelationshipPredictionHead(config)
        if with_mlm_head:
            self.mlm_head = BertOnlyMLMHead(config, self.word_embeddings.weight)
        if with_mvrc_head:
            self.mvrc_head = VisualLinguisticBertMVRCHead(config)

        # --------------------
        # init layer-norm weights
        # --------------------
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_layerNorm_for_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_for_bbox.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.pos_embedding_frozen:
            for p in self.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self,
                txt_token_ids,
                txt_token_type,
                txt_visual_token,
                txt_token_mask,
                bbox_concat_vl_embeddings,
                bbox_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False):
        """
        :param txt_token_ids:
        :param txt_token_type:
        :param txt_visual_token:
        :param txt_token_mask:
        :param bbox_concat_vl_embeddings:
        :param bbox_mask:
        :param output_all_encoded_layers:
        :param output_text_and_object_separately:


        the logic is quite simple:
        1. embedding
        2. head
        based on loss to update embeddings
        """
        # ---------------------------------------------
        # step_1:  call parent cls to obtain embddings
        # two ways to name:
        # 1. embedding
        # 2. encoder
        # ---------------------------------------------
        vl_out_tokens, bbox_out, pooled_rep = super(VisualLinguisticBertForPretraining, self).forward(
            txt_token_ids,
            txt_token_type,
            txt_visual_token,
            txt_token_mask,
            bbox_concat_vl_embeddings,
            bbox_mask,
            output_all_encoded_layers=False,
            output_text_and_object_separately=True
        )


        # ---------------------------------------------
        # step_2: pretraining based on different heads
        # ---------------------------------------------
        if self.with_rel_head:
            relationship_logits = self.relationsip_head(pooled_rep)
        else:
            relationship_logits = None
        if self.with_mlm_head:
            """
            1. we treat tokens, text and viusal, euqally
            2. hidden_dim --> vocab_size
            """
            mlm_logits = self.mlm_head(vl_out_tokens)
        else:
            mlm_logits = None
        if self.with_mvrc_head:
            mvrc_logits = self.mvrc_head(bbox_out)
        else:
            mvrc_logits = None


        # -------------------------------------
        # step_3: return logits from different heads
        # -------------------------------------
        return relationship_logits, mlm_logits, mvrc_logits



    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        relationship_head_pretrained_state_dict = {}
        mlm_head_pretrained_state_dict = {}
        unexpected_keys = []
        for _k, v in pretrained_state_dict.items():
            if _k.startswith('bert.') or _k.startswith('roberta.'):
                k = _k[len('bert.'):] if _k.startswith('bert.') else _k[len('roberta.'):]
                if 'gamma' in k:
                    k = k.replace('gamma', 'weight')
                if 'beta' in k:
                    k = k.replace('beta', 'bias')
                if k.startswith('encoder.'):
                    k_ = k[len('encoder.'):]
                    if k_ in self.bert_encoder.state_dict():
                        encoder_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
                elif k.startswith('embeddings.'):
                    k_ = k[len('embeddings.'):]
                    if k_ == 'word_embeddings.weight':
                        self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                                device=self.word_embeddings.weight.data.device)
                    elif k_ == 'position_embeddings.weight':
                        self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                    device=self.position_embeddings.weight.data.device)
                    elif k_ == 'token_type_embeddings.weight':
                        self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        if v.size(0) == 1:
                            # Todo: roberta token type embedding
                            self.token_type_embeddings.weight.data[1] = v[0].to(
                                dtype=self.token_type_embeddings.weight.data.dtype,
                                device=self.token_type_embeddings.weight.data.device)
                    elif k_.startswith('LayerNorm.'):
                        k__ = k_[len('LayerNorm.'):]
                        if k__ in self.embedding_LayerNorm.state_dict():
                            embedding_ln_pretrained_state_dict[k__] = v
                        else:
                            unexpected_keys.append(_k)
                    else:
                        unexpected_keys.append(_k)
                elif self.config.with_pooler and k.startswith('pooler.'):
                    k_ = k[len('pooler.'):]
                    if k_ in self.pooler.state_dict():
                        pooler_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
            elif _k.startswith('cls.seq_relationship.') and self.with_rel_head:
                k_ = _k[len('cls.seq_relationship.'):]
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.relationsip_head.caption_image_relationship.state_dict():
                    relationship_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            elif (_k.startswith('cls.predictions.') or _k.startswith('lm_head.')) and self.with_mlm_head:
                k_ = _k[len('cls.predictions.'):] if _k.startswith('cls.predictions.') else _k[len('lm_head.'):]
                if _k.startswith('lm_head.'):
                    if 'dense' in k_ or 'layer_norm' in k_:
                        k_ = 'transform.' + k_
                    if 'layer_norm' in k_:
                        k_ = k_.replace('layer_norm', 'LayerNorm')
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.mlm_head.predictions.state_dict():
                    mlm_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            else:
                unexpected_keys.append(_k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.bert_encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)
        if self.with_rel_head and len(relationship_head_pretrained_state_dict) > 0:
            self.relationsip_head.caption_image_relationship.load_state_dict(relationship_head_pretrained_state_dict)
        if self.with_mlm_head:
            self.mlm_head.predictions.load_state_dict(mlm_head_pretrained_state_dict)




class VisualLinguisticBertMVRCHeadTransform(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHeadTransform, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


class VisualLinguisticBertMVRCHead(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHead, self).__init__(config)

        self.transform = VisualLinguisticBertMVRCHeadTransform(config)
        self.region_cls_pred = nn.Linear(config.hidden_size, config.visual_region_classes)
        self.apply(self.init_weights)

    def forward(self, hidden_states):

        hidden_states = self.transform(hidden_states)
        logits = self.region_cls_pred(hidden_states)

        return logits


class VisualLinguisticBertRelationshipPredictionHead(BaseModel):

    def __init__(self, config):
        super(VisualLinguisticBertRelationshipPredictionHead, self).__init__(config)
        self.caption_image_relationship = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, pooled_rep):

        relationship_logits = self.caption_image_relationship(pooled_rep)

        return relationship_logits





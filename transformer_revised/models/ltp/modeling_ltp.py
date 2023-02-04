# coding=utf-8
# Copyright 2021 Samsung Semiconductor Incorporated.
#
# Copyright 2021 The I-BERT Authors (Sehoon Kim, Amir Gholami, Zhewei Yao,
# Michael Mahoney, Kurt Keutzer - UC Berkeley) and The HuggingFace Inc. team.
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch LTP model. """

import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import gelu
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_ltp import LTPConfig
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# from .configuration_ltp import LongformerConfig
# from ..ibert.quant_modules import IntLayerNorm, QuantEmbedding, QuantLinear
from ..longformer.modeling_longformer import LongformerSelfAttention, LongformerAttention, LongformerLayer, LongformerEncoder, LongformerClassificationHead, LongformerEmbeddings, LongformerPooler

from ..longformer.modeling_longformer import LongformerSequenceClassifierOutput
# from ..ibert.modeling_ibert import IBertEmbeddings, IBertSelfAttention,  IBertPooler, IBertLayer, IBertAttention, \
#     IBertEncoder, IBertPreTrainedModel

from .ltp_model_output import LTPModelOutput, LTPSequenceClassifierOutput, LTPEncoderOutput
from .prune_modules import TOKEN_PRUNERS

logger = logging.get_logger(__name__)

# updated

_CONFIG_FOR_DOC = "LTPConfig"
_TOKENIZER_FOR_DOC = "LongformerTokenizer"

LTP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # See all Longformer models at https://huggingface.co/models?filter=longformer
]


class LTPSelfAttention(LongformerSelfAttention):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.prune_mode = config.prune_mode
        self.hard_masking = True
        if self.prune_mode is not None and self.prune_mode in TOKEN_PRUNERS.keys():
            self.pruner = TOKEN_PRUNERS[self.prune_mode](layer_id, **config.prune_kwargs)
        self.module_num = layer_id

    @staticmethod
    def analyse_attention_mask(att_mask, new_att_mask, module_num):
        num_tokens = torch.sum(new_att_mask >= 0)
        num_tokens_before = torch.sum(att_mask >= 0)
        logger.info(f"{num_tokens_before - num_tokens} tokens are pruned in layer {module_num}")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        sentence_lengths=None,
    ):
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # new_attention_mask = attention_mask
        # pruner_outputs = None
        # sentence_len_dict = None
        # add ltp
        # update the attention mask
        pruner_outputs = None
        if self.prune_mode:
            new_attention_mask = self.pruner.update_attention_mask(
                    attention_mask, attn_probs, sentence_lengths,
                    is_global_attn, max_num_global_attn_indices, 
                    is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero)
            if len(new_attention_mask) == 2:
                new_attention_mask, pruner_outputs = new_attention_mask
        else:
            new_attention_mask = attention_mask

        sentence_len_dict = None
        if not self.training:
            # sentence length collection for FLOPs computation
            # attention is computed before token pruning,
            # so the length is computed with pre-pruning attention mask
            attention_mask_binary = attention_mask > -1
            sentence_len_attn = attention_mask_binary.sum(-1).reshape(-1)
            # FFN is computed after token pruning,
            # so the length is computed with post-pruning attention mask
            new_attention_mask_binary = new_attention_mask > -1
            sentence_len_ffn = new_attention_mask_binary.sum(-1).reshape(-1)
            sentence_len_dict = {'attn': sentence_len_attn, 'ffn': sentence_len_ffn}

        self.analyse_attention_mask(attention_mask, new_attention_mask, self.module_num)

        # If training with the soft masking mode, do not set attention mask
        # Instead, soft masking will be applied at the end of each layer
        if self.training and not self.hard_masking:
            new_attention_mask = attention_mask

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs, new_attention_mask, pruner_outputs, sentence_len_dict


class LTPAttention(LongformerAttention):
    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id=layer_id)
        self.self = LTPSelfAttention(config, layer_id)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        sentence_lengths=None,
    ):
        self_outputs, new_attention_mask, \
        pruner_outputs, sentence_len_dict = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
            sentence_lengths=sentence_lengths,
        )
        attention_output = self.output(
            self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, new_attention_mask, pruner_outputs, sentence_len_dict


class LTPLayer(LongformerLayer):
    def __init__(self, config, module_num):
        super().__init__(config, module_num)
        self.attention = LTPAttention(config, module_num)
        self.module_num = module_num
        self.mask = None
        self.hard_masking = False
        self.temperature = 1e-5

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        sentence_lengths=None,
    ):
        self_attention_outputs, \
        new_attention_mask, pruner_outputs, sentence_len_dict = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
            sentence_lengths=sentence_lengths,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.training and not self.hard_masking:
            if pruner_outputs is not None:
                threshold, pruning_scores = pruner_outputs['threshold'], pruner_outputs['scores']
                self.mask = torch.sigmoid((pruning_scores - threshold) / self.temperature)
                layer_output = layer_output * self.mask.unsqueeze(-1)
            
        outputs = (layer_output,) + outputs

        return outputs, new_attention_mask, sentence_len_dict

# updated
class LTPEncoder(LongformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([LTPLayer(config, i) for i in range(config.num_hidden_layers)])
        self.prune_mode = config.prune_mode

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        all_hidden_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None  # `config.add_cross_attention` is not supported
        next_decoder_cache = None  # `config.use_cache` is not supported

        batch_size = attention_mask.shape[0]
        sentence_lengths = (attention_mask == 0).view(batch_size, -1).sum(dim=1) if self.prune_mode else None

        attention_sentence_lengths = [] # per layer, store list for each example in batch
        ffn_sentence_lengths = [] # per layer, store list for each example in batch

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                raise NotImplementedError("gradient checkpointing is not currently supported")

            else:
                layer_outputs, new_attention_mask, sentence_len_dict = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    is_index_masked,
                    is_index_global_attn,
                    is_global_attn,
                    output_attentions,
                    sentence_lengths,
                )
            hidden_states = layer_outputs[0]

            attention_mask = new_attention_mask

            if sentence_len_dict is not None:
                attention_sentence_lengths.append(sentence_len_dict['attn'])
                ffn_sentence_lengths.append(sentence_len_dict['ffn'])

            if output_attentions:
                # # updated
                # import pdb
                # pdb.set_trace()
                all_self_attentions = all_self_attentions + (layer_outputs[1:-1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                all_hidden_states = tuple([state[:, :-padding_len] for state in all_hidden_states])

            if output_attentions:
                all_self_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_global_attentions,
                    all_cross_attentions,
                    attention_mask,
                    attention_sentence_lengths,
                    ffn_sentence_lengths,
                ]
                if v is not None
            )
        return LTPEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            global_attentions=all_global_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            attention_sentence_lengths=attention_sentence_lengths,
            ffn_sentence_lengths=ffn_sentence_lengths,
        )


class LTPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LTPConfig
    base_model_prefix = "longformer"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LongformerEncoder):
            # module.gradient_checkpointing = value
            assert value == False

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError("`resize_token_embeddings` is not supported for LTP.")


LTP_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.LTPConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LTP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.RobertaTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LTP Model transformer outputting raw hidden-states without any specific head on top.",
    LTP_START_DOCSTRING,
)
class LTPModel(LTPPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LTPEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        self.post_init()

        self.hard_masking = False

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def set_temperature(self, temperature):
        for layer in self.encoder.layer:
            layer.temperature = temperature

    def set_hard_masking(self, hard_masking: bool):
        for layer in self.encoder.layer:
            layer.attention.self.hard_masking = hard_masking
            layer.hard_masking = hard_masking

    def get_soft_mask(self):
        return [layer.mask for layer in self.encoder.layer]

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,  # 
        checkpoint="ltp-roberta-base",
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[:, 0, 0, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LTPModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            global_attentions=encoder_outputs.global_attentions,
            attention_mask=encoder_outputs.attention_mask,
            attention_sentence_lengths=encoder_outputs.attention_sentence_lengths,
            ffn_sentence_lengths=encoder_outputs.ffn_sentence_lengths,
        )


# @add_start_docstrings("""LTP Model with a `language modeling` head on top. """, LTP_START_DOCSTRING)
# class LTPForMaskedLM(LTPPreTrainedModel):
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
# 
#     def __init__(self, config):
#         super().__init__(config)
# 
#         self.ibert = LTPModel(config, add_pooling_layer=False)
#         self.lm_head = LTPLMHead(config)
# 
#         self.init_weights()
# 
#     def get_output_embeddings(self):
#         return self.lm_head.decoder
# 
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head.decoder = new_embeddings
# 
#     @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         processor_class=_TOKENIZER_FOR_DOC,
#         checkpoint="ltp-roberta-base",
#         output_type=MaskedLMOutput,
#         config_class=_CONFIG_FOR_DOC,
#         mask="<mask>",
#     )
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
#             config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
#             (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
#         kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
#             Used to hide legacy arguments that have been deprecated.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
# 
#         outputs = self.ibert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         prediction_scores = self.lm_head(sequence_output)
# 
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
# 
#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
# 
#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
# 
# 
# class LTPLMHead(nn.Module):
#     """LTP Head for masked language modeling."""
# 
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
# 
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
# 
#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias
# 
#     def forward(self, features, **kwargs):
#         x = self.dense(features)
#         x = gelu(x)
#         x = self.layer_norm(x)
# 
#         # project back to size of vocabulary with bias
#         x = self.decoder(x)
# 
#         return x


@add_start_docstrings(
    """
    LTP Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    LTP_START_DOCSTRING,
)
class LTPForSequenceClassification(LTPPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LTPModel(config, add_pooling_layer=False)
        self.classifier = LTPClassificationHead(config)

        self.post_init()

    @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="ltp-roberta-base",
        output_type=LongformerSequenceClassifierOutput,  # revised
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 2` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            # logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
            # updated

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # updated
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LTPSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
            attention_sentence_lengths=outputs.attention_sentence_lengths,
            ffn_sentence_lengths=outputs.ffn_sentence_lengths,
        )

    def get_model(self):
        return self.longformer

    def get_rest_token_indices(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            # logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
            # updated

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 
        sentence_lengths = outputs.attention_sentence_lengths[0]
        attention_mask = outputs.attention_mask >= 0
        batch_size = attention_mask.shape[0]
        rest_indices = []
        for i in range(batch_size):
            row = attention_mask[i]
            rest_indices.append(row.nonzero().squeeze(-1))
        return rest_indices, sentence_lengths


@add_start_docstrings(
    """
    LTP Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    LTP_START_DOCSTRING,
)
class LTPForMultipleChoice(LTPPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.ibert = LTPModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="ltp-roberta-base",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.ibert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    LTP Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    LTP_START_DOCSTRING,
)
class LTPForTokenClassification(LTPPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ibert = LTPModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="ltp-roberta-base",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ibert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LTPClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        hidden_states = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    LTP Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LTP_START_DOCSTRING,
)
class LTPForQuestionAnswering(LTPPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ibert = LTPModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LTP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="ltp-roberta-base",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ibert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
    input_ids (:obj:`torch.LongTensor`):
           Indices of input sequence tokens in the vocabulary.

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

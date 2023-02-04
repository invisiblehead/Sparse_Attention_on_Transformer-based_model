# Copyright 2021 Samsung Semiconductor Incorporated.

import torch
import torch.nn as nn
import math

from ...utils import logging

logger = logging.get_logger(__name__)

# Only AbsoluteTokenPruner is revised

class AbstractTokenPruner(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths):
        return attention_mask


class CascadeTokenPruner(AbstractTokenPruner):
    """
    implements the layer-by-layer operations for the cascade token pruning method described in:

    Wang et al.,
    SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning
    https://arxiv.org/abs/2012.09852
    """
    def __init__(self, module_num, token_keep_rate, num_hidden_layers, **kwargs):
        super().__init__()
        self.keep_rate = self._set_token_keep_rate(module_num, num_hidden_layers, token_keep_rate)
        self.threshold_score = None

    @staticmethod
    def _set_token_keep_rate(i, num_hidden_layers, token_keep_rate):
        """
        Following the SpAtten paper, the rules for token pruning are:
        * the first 3 or 15% of layers, whichever is greater, should not be token pruned
        * for the remaining layers, the fraction of pruned tokens should increase linearly until the desired final
        value is reached.
        This method implements these rules and sets the keep_rate field for each pruner in each LTPLayer.
        """
        layers_before_pruning = max(3, math.ceil(0.15 * num_hidden_layers))
        layers_with_pruning = num_hidden_layers - layers_before_pruning
        if i < layers_before_pruning:
            return 1.0
        else:
            m = (token_keep_rate - 1) / layers_with_pruning
            tkr =  m * (i - layers_before_pruning + 1) + 1
            tkr = max(0.01, tkr)
            logger.info(f"Layer {i} token keep rate: {tkr}")
            return tkr

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths):
        keep_tokens = torch.round(sentence_lengths * self.keep_rate).long()
        sz = attention_probs.shape[-1]
        batch_size = attention_mask.shape[0]
        self.threshold_score = torch.zeros((batch_size,))
        if self.keep_rate == 1:
            return attention_mask

        # compute the pruning scores by summing the attention probabilities over all heads
        attention_mask_index = (attention_mask < 0).permute(0, 1, 3, 2).repeat(1, attention_probs.shape[1], 1, sz)
        attention_probs[attention_mask_index] = 0
        pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)

        # sort the pruning scores using the top-k engine
        top_scores, sorted_indices = torch.sort(-pruning_scores, dim=-1)

        # construct the new attention mask
        new_attention_mask = torch.ones(attention_mask.shape, device=attention_mask.device) * -10000
        # TODO: remove for loop if possible
        for i in range(batch_size):
            new_attention_mask[i, ..., sorted_indices[i, 0:keep_tokens[i]]] = 0
            # if keep_tokens[i] < sz:
            #    self.threshold_score[i] = -top_scores[i, keep_tokens[i]] / torch.max(-top_scores[i, ...])

        return new_attention_mask

# I think this one is the best
class ThresholdTokenPruner(AbstractTokenPruner):
    """
    implements the layer-by-layer operations for threshold token pruning, where tokens are pruned if the importance
    score is strictly less than a given fraction of the maximum token importance score
    """
    def __init__(self, module_num, token_threshold, **kwargs):
        super().__init__()
        self.keep_threshold = token_threshold

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths):
        sz = attention_probs.shape[-1]
        batch_size = attention_mask.shape[0]
        if self.keep_threshold == 0:
            return attention_mask

        # compute the pruning scores by summing the attention probabilities over all heads
        attention_mask_index = (attention_mask < 0).permute(0, 1, 3, 2).repeat(1, attention_probs.shape[1], 1, sz)
        attention_probs[attention_mask_index] = 0
        pruning_scores = attention_probs.view(batch_size, -1, sz).sum(dim=1)

        max_pruning_scores, _ = torch.max(pruning_scores, dim=-1, keepdim=True)
        relative_pruning_scores = pruning_scores / max_pruning_scores

        # construct the new attention mask
        new_attention_mask = torch.zeros(attention_mask.shape, device=attention_mask.device)
        new_attention_mask[relative_pruning_scores.unsqueeze(1).unsqueeze(1) < self.keep_threshold] = -10000

        return new_attention_mask


class RisingThresholdTokenPruner(ThresholdTokenPruner):
    def __init__(self, module_num, final_token_threshold=None, num_hidden_layers=None, **kwargs):
        super().__init__()
        self.keep_threshold = final_token_threshold * module_num / num_hidden_layers


class AbsoluteThresholdTokenPruner(AbstractTokenPruner):
    """
    implements the layer-by-layer operations for threshold token pruning, where tokens are pruned if the importance
    score is strictly less than a given fraction of the maximum token importance score
    """
    def __init__(self, module_num, final_token_threshold=None, num_hidden_layers=None, **kwargs):
        super().__init__()
        self.keep_threshold_base = torch.tensor(final_token_threshold * module_num / num_hidden_layers, device='cuda')
        self.keep_threshold = nn.Parameter(
                torch.zeros_like(self.keep_threshold_base,  device='cuda'),
                requires_grad=True,
        )
        self.module_num = module_num

        logger.info("Layer %d Threshold: %f" % (module_num, float(self.keep_threshold_base + self.keep_threshold)))

    def update_attention_mask(self, attention_mask, attention_probs, sentence_lengths, is_global_attn, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        # attention_probs
        attention_probs = attention_probs.clone().detach()

        # if threshold is zero, simply do not update
        keep_threshold = self.keep_threshold + self.keep_threshold_base
        if keep_threshold == 0:
            return attention_mask

        # notation: B batch_size, L seq_len, H num_head, G #global, W #window_size (#local), R=G+W
        # attention_probs (B, L, H=#head, R=#global_atten+window_size)
        # firstly we mask attention_probs by zeroing padded tokens along L
        # attention_mask (B, L) -> attention_mask_index (B, L=128, H, W)
        batch_size, seq_len, num_head, window_size_and_global = attention_probs.shape
        attention_mask_index = (attention_mask < 0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_head, window_size_and_global)
        attention_probs[attention_mask_index] = 0 

        # split attention_probs into global_attention_probs and local_attention_probs
        global_attn_probs = attention_probs.narrow(-1, 0, max_num_global_attn_indices)
        local_attn_probs = attention_probs.narrow(-1, max_num_global_attn_indices, attention_probs.size(-1) - max_num_global_attn_indices).contiguous()

        # compute pruner scores from local_attn_probs at first
        window_size = local_attn_probs.shape[-1]
        assert window_size % 2 == 1
        # now we compute the pruning scores from local_attn_probs 
        #   by summing the attention probabilities over all heads
        # local_attn_probs is of shape (B, L, H, W), we want derive pruning_scores (B, L) from it
        # however a simple sum is incorrect for local_attn_probs generated from longformer
        # we need the sum of counter-off-diagonal items from local_attn_probs
        # flt shape: (num_head, 1, window_size, window_size)
        flt = torch.eye(window_size, device=attention_mask.device)[None, None,...].flip(-1).repeat(num_head, 1, 1, 1)
        # counter_diag_sums shape: (B, H, L+window_size-1)
        counter_diag_sums = nn.functional.conv2d(local_attn_probs.transpose(1, 2), flt, padding=(window_size-1,0), groups=num_head)[..., 0]
        # remove the edge items that are useless
        # counter_diag_sums_corrected: (B, H, L)
        counter_diag_sums_corrected = counter_diag_sums[:, :, window_size//2:-window_size//2+1]
        assert counter_diag_sums_corrected.shape[-1] == seq_len
        # (B, L)
        local_probs_sum = counter_diag_sums_corrected.sum(dim=1)

        # compute pruner scores from global_attn_probs
        # global_attn_probs: (B, L, H, G) -> global_probs_sum: (B, G)
        global_probs_sum = global_attn_probs.sum(dim=(1, 2))

        # add global_probs_sum into local_probs_sum according to is_index_global_attn_nonzero
        probs_sum = local_probs_sum
        probs_sum[is_local_index_global_attn_nonzero] += global_probs_sum[is_index_global_attn_nonzero]
        
        # normalize probs_sums and get pruning_scores
        probs_max =  torch.max(probs_sum, dim=-1)[0].unsqueeze(-1)
        pruning_scores = probs_sum / probs_max

        # update the mask
        new_attention_mask = torch.zeros(attention_mask.shape, device=attention_mask.device)
        new_attention_mask[pruning_scores < max(1e-5, keep_threshold)] = -10000
        pruner_outputs = {'threshold': keep_threshold, 'scores': pruning_scores}

        num_tokens = torch.sum(new_attention_mask >= 0)
        num_tokens_before = torch.sum(attention_mask >= 0)
        print(f"{num_tokens_before - num_tokens} tokens are pruned, {num_tokens} left")

        return new_attention_mask, pruner_outputs


TOKEN_PRUNERS = {'topk': CascadeTokenPruner,
                 'threshold': ThresholdTokenPruner,
                 'rising_threshold': RisingThresholdTokenPruner,
                 'absolute_threshold': AbsoluteThresholdTokenPruner,
                 }


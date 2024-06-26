:py:mod:`lmflow.utils.flash_attention.bloom_flash_attention`
============================================================

.. py:module:: lmflow.utils.flash_attention.bloom_flash_attention


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.bloom_flash_attention.forward
   lmflow.utils.flash_attention.bloom_flash_attention._prepare_attn_mask
   lmflow.utils.flash_attention.bloom_flash_attention.replace_bloom_attn_with_flash_attn



.. py:function:: forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, head_mask: Optional[torch.Tensor] = None, use_cache: bool = False, output_attentions: bool = False)


.. py:function:: _prepare_attn_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int) -> torch.BoolTensor


.. py:function:: replace_bloom_attn_with_flash_attn()



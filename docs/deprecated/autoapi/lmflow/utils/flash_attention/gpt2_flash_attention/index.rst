:py:mod:`lmflow.utils.flash_attention.gpt2_flash_attention`
===========================================================

.. py:module:: lmflow.utils.flash_attention.gpt2_flash_attention


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.gpt2_flash_attention.forward
   lmflow.utils.flash_attention.gpt2_flash_attention._prepare_decoder_attention_mask
   lmflow.utils.flash_attention.gpt2_flash_attention.replace_gpt2_attn_with_flash_attn



.. py:function:: forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]] = None, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = False, output_attentions: Optional[bool] = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], Ellipsis]


.. py:function:: _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)


.. py:function:: replace_gpt2_attn_with_flash_attn()



:py:mod:`lmflow.utils.flash_attention.gpt_neo_flash_attention`
==============================================================

.. py:module:: lmflow.utils.flash_attention.gpt_neo_flash_attention


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.gpt_neo_flash_attention._attn
   lmflow.utils.flash_attention.gpt_neo_flash_attention.forward
   lmflow.utils.flash_attention.gpt_neo_flash_attention.replace_gpt_neo_attn_with_flash_attn



.. py:function:: _attn(self, query, key, value, attention_mask=None, head_mask=None)


.. py:function:: forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False)


.. py:function:: replace_gpt_neo_attn_with_flash_attn()



:py:mod:`lmflow.datasets.multi_modal_dataset`
=============================================

.. py:module:: lmflow.datasets.multi_modal_dataset

.. autoapi-nested-parse::

   This Python code defines a class Multi Modal Dataset.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.datasets.multi_modal_dataset.CustomMultiModalDataset
   lmflow.datasets.multi_modal_dataset.DataCollatorForSupervisedDataset



Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.datasets.multi_modal_dataset.preprocess_multimodal_llava
   lmflow.datasets.multi_modal_dataset.tokenizer_image_token
   lmflow.datasets.multi_modal_dataset.preprocess_llama_from_llava_plain
   lmflow.datasets.multi_modal_dataset.preprocess_llama_from_llava_v1



.. py:class:: CustomMultiModalDataset(dataset_path: str, data_args: lmflow.args.DatasetArguments)

   Bases: :py:obj:`torch.utils.data.Dataset`

   
   Dataset for Multi Modal data
















   ..
       !! processed by numpydoc !!
   .. py:method:: __len__()


   .. py:method:: register_tokenizer(tokenizer, image_processor=None)


   .. py:method:: __getitem__(i)



.. py:function:: preprocess_multimodal_llava(sources, data_args)


.. py:function:: tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None)


.. py:function:: preprocess_llama_from_llava_plain(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False)

   
   This function just add the image in the front of text.
   And don't add any prompt.
   Args:
       sources: The input data with text and image.
       tokenizer: The tokenizer to process text.
       has_image: Whether the input data has image.
   Returns:
       The input_ids and labels for the model.
















   ..
       !! processed by numpydoc !!

.. py:function:: preprocess_llama_from_llava_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False)

   
   This function add the prompt and then put the image after the prompt.
   So it needs additional code to generate the target label.
   Args:
       sources: The input data with text and image.
       tokenizer: The tokenizer to process text.
       has_image: Whether the input data has image.
   Returns:
       The input_ids and labels for the model.
















   ..
       !! processed by numpydoc !!

.. py:class:: DataCollatorForSupervisedDataset

   Bases: :py:obj:`object`

   
   Collate examples for supervised fine-tuning.
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: tokenizer
      :annotation: :transformers.PreTrainedTokenizer

      

   .. py:method:: __call__(instances)




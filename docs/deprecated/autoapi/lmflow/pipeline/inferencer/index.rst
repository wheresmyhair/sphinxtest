:py:mod:`lmflow.pipeline.inferencer`
====================================

.. py:module:: lmflow.pipeline.inferencer

.. autoapi-nested-parse::

   The Inferencer class simplifies the process of model inferencing.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.inferencer.Inferencer
   lmflow.pipeline.inferencer.SpeculativeInferencer



Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.pipeline.inferencer.rstrip_partial_utf8



Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.pipeline.inferencer.supported_dataset_type
   lmflow.pipeline.inferencer.logger


.. py:function:: rstrip_partial_utf8(string)


.. py:data:: supported_dataset_type
   :annotation: = ['text_only', 'image_text']

   

.. py:data:: logger
   

   

.. py:class:: Inferencer(model_args, data_args, inferencer_args)

   Bases: :py:obj:`lmflow.pipeline.base_pipeline.BasePipeline`

   
   Initializes the `Inferencer` class with given arguments.


   :Parameters:

       **model_args** : ModelArguments object.
           Contains the arguments required to load the model.

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **inferencer_args** : InferencerArguments object.
           Contains the arguments required to perform inference.














   ..
       !! processed by numpydoc !!
   .. py:method:: create_dataloader(dataset: lmflow.datasets.dataset.Dataset)

      
      Batchlize dataset and format it to dataloader.

      Args:
          dataset (Dataset): the dataset object

      Output:
          dataloader (batchlize): the dataloader object
          dataset_size (int): the length of the dataset















      ..
          !! processed by numpydoc !!

   .. py:method:: inference(model, dataset: lmflow.datasets.dataset.Dataset, max_new_tokens: int = 100, temperature: float = 0.0, prompt_structure: str = '{input}', remove_image_flag: bool = False, chatbot_type: str = 'mini_gpt')

      
      Perform inference for a model


      :Parameters:

          **model** : TunableModel object.
              TunableModel to perform inference

          **dataset** : Dataset object.
              ..

          **Returns:**
              ..

          **output_dataset: Dataset object.**
              ..














      ..
          !! processed by numpydoc !!

   .. py:method:: stream_inference(context, model, max_new_tokens, token_per_step, temperature, end_string, input_dataset, remove_image_flag: bool = False)



.. py:class:: SpeculativeInferencer(model_args, draft_model_args, data_args, inferencer_args)

   Bases: :py:obj:`Inferencer`

   
   Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192)


   :Parameters:

       **target_model_args** : ModelArguments object.
           Contains the arguments required to load the target model.

       **draft_model_args** : ModelArguments object.
           Contains the arguments required to load the draft model.

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **inferencer_args** : InferencerArguments object.
           Contains the arguments required to perform inference.














   ..
       !! processed by numpydoc !!
   .. py:method:: score_to_prob(scores: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0) -> torch.Tensor
      :staticmethod:

      
      Convert scores (NOT softmaxed tensor) to probabilities with support for temperature, top-p sampling, and argmax.


      :Parameters:

          **scores** : torch.Tensor
              Input scores.

          **temperature** : float, optional
              Temperature parameter for controlling randomness. Higher values make the distribution more uniform, 
              lower values make it peakier. When temperature <= 1e-6, argmax is used. by default 0.0

          **top_p** : float, optional
              Top-p sampling parameter for controlling the cumulative probability threshold, by default 1.0 (no threshold)

      :Returns:

          torch.Tensor
              Probability distribution after adjustments.













      ..
          !! processed by numpydoc !!

   .. py:method:: sample(prob: torch.Tensor, num_samples: int = 1) -> Dict
      :staticmethod:

      
      Sample from a tensor of probabilities
















      ..
          !! processed by numpydoc !!

   .. py:method:: predict_next_token(model: lmflow.models.hf_decoder_model.HFDecoderModel, input_ids: torch.Tensor, num_new_tokens: int = 1)
      :staticmethod:

      
      Predict the next token given the input_ids.
















      ..
          !! processed by numpydoc !!

   .. py:method:: autoregressive_sampling(input_ids: torch.Tensor, model: lmflow.models.hf_decoder_model.HFDecoderModel, temperature: float = 0.0, num_new_tokens: int = 5) -> Dict

      
      Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192) Section 2.2
















      ..
          !! processed by numpydoc !!

   .. py:method:: inference(model: lmflow.models.hf_decoder_model.HFDecoderModel, draft_model: lmflow.models.hf_decoder_model.HFDecoderModel, input: str, temperature: float = 0.0, gamma: int = 5, max_new_tokens: int = 100)

      
      Perform inference for a model


      :Parameters:

          **model** : HFDecoderModel object.
              TunableModel to verify tokens generated by the draft model.

          **draft_model** : HFDecoderModel object.
              TunableModel that provides approximations of the target model.

          **input** : str.
              The input text (i.e., the prompt) for the model.

          **gamma** : int.
              The number of tokens to be generated by the draft model within each iter.

          **max_new_tokens** : int.
              The maximum number of tokens to be generated by the target model.

      :Returns:

          output: str.
              The output text generated by the model.













      ..
          !! processed by numpydoc !!

   .. py:method:: stream_inference()
      :abstractmethod:




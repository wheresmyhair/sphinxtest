:py:mod:`lmflow.models.hf_encoder_decoder_model`
================================================

.. py:module:: lmflow.models.hf_encoder_decoder_model

.. autoapi-nested-parse::

   This is a class called HFDecoderModel which is a wrapper around transformers model and
   tokenizer classes. It has several methods such as __init__, tokenize, and train that are
   used for training and fine-tuning the model. The __init__ method takes in several arguments
   such as model_args, tune_strategy, and ds_config, which are used to load the pretrained
   model and tokenizer, and initialize the training settings.

   The tokenize method is used to tokenize the input text and return the input IDs and attention
   masks that can be fed to the model for training or inference.

   This class supports different tune_strategy options such as 'normal', 'none', 'lora', and
   'adapter', which allow for different fine-tuning settings of the model. However, the 'lora'
   and 'adapter' strategies are not yet implemented.

   Overall, this class provides a convenient interface for loading and fine-tuning transformer
   models and can be used for various NLP tasks such as language modeling, text classification,
   and question answering.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.models.hf_encoder_decoder_model.HFEncoderDecoderModel




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.models.hf_encoder_decoder_model.logger


.. py:data:: logger
   

   

.. py:class:: HFEncoderDecoderModel(model_args, tune_strategy='normal', ds_config=None, device='gpu', use_accelerator=False, custom_model=False, with_deepspeed=True, pipeline_args=None, *args, **kwargs)

   Bases: :py:obj:`lmflow.models.encoder_decoder_model.EncoderDecoderModel`, :py:obj:`lmflow.models.interfaces.tunable.Tunable`

   
   Initializes a HFEncoderDecoderModel instance.


   :Parameters:

       **model_args :**
           Model arguments such as model name, path, revision, etc.

       **tune_strategy** : str or none,  default="normal".
           A string representing the dataset backend. Defaults to "huggingface".

       **ds_config :**
           Deepspeed configuations.

       **args** : Optional.
           Positional arguments.

       **kwargs** : Optional.
           Keyword arguments.














   ..
       !! processed by numpydoc !!
   .. py:method:: tokenize(dataset, *args, **kwargs)
      :abstractmethod:

      
      Tokenize the full dataset.


      :Parameters:

          **dataset :**
              Text dataset.

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          tokenized_datasets :
              The tokenized dataset.













      ..
          !! processed by numpydoc !!

   .. py:method:: encode(input: Union[str, List[str]], *args, **kwargs) -> Union[List[int], List[List[int]]]

      
      Perform encoding process of the tokenizer.


      :Parameters:

          **inputs** : str or list.
              The text sequence.

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          outputs :
              The tokenized inputs.













      ..
          !! processed by numpydoc !!

   .. py:method:: decode(input, *args, **kwargs) -> Union[str, List[str]]

      
      Perform decoding process of the tokenizer.


      :Parameters:

          **inputs** : list.
              The token sequence.

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          outputs :
              The text decoded from the token inputs.













      ..
          !! processed by numpydoc !!

   .. py:method:: inference(inputs, *args, **kwargs)

      
      Perform generation process of the model.


      :Parameters:

          **inputs :**
              The sequence used as a prompt for the generation or as model inputs to the model.

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          outputs :
              The generated sequence output













      ..
          !! processed by numpydoc !!

   .. py:method:: merge_lora_weights()


   .. py:method:: save(dir, save_full_model=False, *args, **kwargs)

      
      Perform generation process of the model.


      :Parameters:

          **dir :**
              The directory to save model and tokenizer

          **save_full_model** : Optional.
              Whether to save full model.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          outputs :
              The generated sequence output













      ..
          !! processed by numpydoc !!

   .. py:method:: get_max_length()

      
      Return max acceptable input length in terms of tokens.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_tokenizer()

      
      Return the tokenizer of the model.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_backend_model()

      
      Return the backend model.
















      ..
          !! processed by numpydoc !!



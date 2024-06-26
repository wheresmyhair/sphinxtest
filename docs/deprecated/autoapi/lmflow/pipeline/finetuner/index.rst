:py:mod:`lmflow.pipeline.finetuner`
===================================

.. py:module:: lmflow.pipeline.finetuner

.. autoapi-nested-parse::

   The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.finetuner.Finetuner




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.pipeline.finetuner.logger


.. py:data:: logger
   

   

.. py:class:: Finetuner(model_args, data_args, finetuner_args, *args, **kwargs)

   Bases: :py:obj:`lmflow.pipeline.base_tuner.BaseTuner`

   
   Initializes the `Finetuner` class with given arguments.


   :Parameters:

       **model_args** : ModelArguments object.
           Contains the arguments required to load the model.

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **finetuner_args** : FinetunerArguments object.
           Contains the arguments required to perform finetuning.

       **args** : Optional.
           Positional arguments.

       **kwargs** : Optional.
           Keyword arguments.














   ..
       !! processed by numpydoc !!
   .. py:method:: group_text(tokenized_datasets, model_max_length)

      
      Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
      a dictionary.
















      ..
          !! processed by numpydoc !!

   .. py:method:: tune(model, dataset, transform_dataset_in_place=True, data_collator=None)

      
      Perform tuning for a model


      :Parameters:

          **model** : TunableModel object.
              TunableModel to perform tuning.

          **dataset:**
              dataset to train model.














      ..
          !! processed by numpydoc !!



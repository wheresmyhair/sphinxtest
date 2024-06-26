:py:mod:`lmflow.pipeline.raft_aligner`
======================================

.. py:module:: lmflow.pipeline.raft_aligner

.. autoapi-nested-parse::

   The Aligner class simplifies the process of running alignment.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.raft_aligner.RaftAligner




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.pipeline.raft_aligner.logger


.. py:data:: logger
   

   

.. py:class:: RaftAligner(model_args, data_args, aligner_args, *args, **kwargs)

   Bases: :py:obj:`lmflow.pipeline.base_aligner.BaseAligner`

   
   Initializes the `RaftAligner` class with given arguments.


   :Parameters:

       **model_args** : ModelArguments object.
           Contains the arguments required to load the model.

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **raft_aligner_args** : RaftAlignerArguments object.
           Contains the arguments required to perform alignment.

       **args** : Optional.
           Positional arguments.

       **kwargs** : Optional.
           Keyword arguments.














   ..
       !! processed by numpydoc !!
   .. py:method:: _initialize_trainer(model, tokenizer, training_args)

      
      This function takes the model and tokenizer as the input and initialize the trainer.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _load_dataset(selected_dataset, model, tokenizer, model_args, data_args, training_args)

      
      This function prepares the dataset for every iteration.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _load_input_dataset(dataset, tokenizer)

      
      Load input dataset (i.e. prompt/question dataset) for training.

      Args:
          dataset: A Dataset object.
              The dataset to be loaded.

      Returns:
          dataloader (`torch.utils.data.DataLoader`):
              The dataloader for the dataset.















      ..
          !! processed by numpydoc !!

   .. py:method:: _clean_text(text)


   .. py:method:: _discard_sample(text)


   .. py:method:: _get_batch_dataset_top(model, batch_input, alpha=0.2, iter_id=0, local_rank=0, output_min_length=16, output_max_length=48, infer_batch_size=8, generation_kwargs={}, tokenizer=None, training_args=None, reward_model=None, output_reward_path=None)

      
      :param batch_input: input prompts
















      ..
          !! processed by numpydoc !!

   .. py:method:: _get_batch_dataset_local(model, batch_input, K=8, iter_id=0, local_rank=0, output_min_length=16, output_max_length=48, infer_batch_size=8, generation_kwargs={}, tokenizer=None, training_args=None, reward_model=None, output_reward_path=None)

      
      :param batch_input: input prompts
















      ..
          !! processed by numpydoc !!

   .. py:method:: align(model, dataset, reward_model)

      
      Perform alignment for a model


      :Parameters:

          **model** : BaseModel object.
              ..

          **dataset: Dataset object.**
              Input dataset for model to generate outputs. The input and output
                  will then be feed into reward model to get the reward for
                  alignment.

          **reward_model: RegressionModel object.**
              ..














      ..
          !! processed by numpydoc !!



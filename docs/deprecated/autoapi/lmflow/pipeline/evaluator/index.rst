:py:mod:`lmflow.pipeline.evaluator`
===================================

.. py:module:: lmflow.pipeline.evaluator

.. autoapi-nested-parse::

   The Evaluator class simplifies the process of running evaluation on a language model provided by a HFDecoderModel instance imported from the lmflow package. The class constructor takes three dictionaries as arguments: model_args containing arguments related to the language model, data_args containing arguments related to the data used for evaluation, and evaluator_args containing other arguments for the evaluation process.

   The class has two methods: create_dataloader() that loads the data from the test file, creates a data loader, and returns it with the size of the data, and evaluate(model) that generates output text given input text. It uses the create_dataloader() method to load the data, iterates over the data in mini-batches, and encodes the input text with the encode() method of the HFDecoderModel class. Then, it generates output text using the evaluate() method of the HFDecoderModel class, decodes the generated output text using the decode() method of the HFDecoderModel class, and writes the output to a file in the output directory. The method also logs some information to the console and Weights and Biases if the use_wandb argument is True.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.evaluator.Evaluator




.. py:class:: Evaluator(model_args, data_args, evaluator_args)

   Bases: :py:obj:`lmflow.pipeline.base_pipeline.BasePipeline`

   
   Initializes the `Evaluator` class with given arguments.


   :Parameters:

       **model_args** : ModelArguments object.
           Contains the arguments required to load the model.

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **evaluator_args** : EvaluatorArguments object.
           Contains the arguments required to perform evaluation.














   ..
       !! processed by numpydoc !!
   .. py:method:: create_dataloader(dataset: lmflow.datasets.dataset.Dataset)


   .. py:method:: _match(predicted_answer, groundtruth, answer_type=None)


   .. py:method:: evaluate(model, dataset: lmflow.datasets.dataset.Dataset, metric='accuracy', verbose=True)

      
      Perform Evaluation for a model


      :Parameters:

          **model** : TunableModel object.
              TunableModel to perform inference

          **dataset** : Dataset object.
              ..














      ..
          !! processed by numpydoc !!

   .. py:method:: _evaluate_acc_with_accelerator(model, dataset, verbose=True)


   .. py:method:: _evaluate_acc_with_deepspeed(model, dataset, verbose=True)


   .. py:method:: _evaluate_ppl(model, dataset: lmflow.datasets.dataset.Dataset, verbose=True)


   .. py:method:: _evaluate_nll(model, dataset: lmflow.datasets.dataset.Dataset, verbose=True)

      
      Evaluates negative log likelihood of the model over a dataset.

      NLL = -1/N sum_{i=1}^N sum_{j=1}^|w_i| ln(p(w_{i,j}|context_window)),

      where N is the number of data samples, w_{i,j} is the j-th token in
      i-th sample. Here "context_window" = p(w_{i,start}, w_{i,start+1}, ...,
      p_{i,j-1} with start = max(0, j - window_length + 1). "window_length"
      is normally the maximum length accepted by the model.

      Returns:
          A float which represents the negative log likelihood.















      ..
          !! processed by numpydoc !!



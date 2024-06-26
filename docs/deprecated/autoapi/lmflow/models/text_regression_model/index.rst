:py:mod:`lmflow.models.text_regression_model`
=============================================

.. py:module:: lmflow.models.text_regression_model

.. autoapi-nested-parse::

   A model maps "text_only" data to float.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.models.text_regression_model.TextRegressionModel




.. py:class:: TextRegressionModel(model_args, *args, **kwargs)

   Bases: :py:obj:`lmflow.models.regression_model.RegressionModel`

   
   Initializes a TextRegressionModel instance.


   :Parameters:

       **model_args :**
           Model arguments such as model name, path, revision, etc.

       **args** : Optional.
           Positional arguments.

       **kwargs** : Optional.
           Keyword arguments.    














   ..
       !! processed by numpydoc !!
   .. py:method:: register_inference_function(inference_func)

      
      Registers a regression function.
















      ..
          !! processed by numpydoc !!

   .. py:method:: inference(inputs: lmflow.datasets.dataset.Dataset)

      
      Gets regression results of a given dataset.

      :inputs: Dataset object, only accept type "text_only".















      ..
          !! processed by numpydoc !!



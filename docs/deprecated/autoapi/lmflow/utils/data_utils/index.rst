:py:mod:`lmflow.utils.data_utils`
=================================

.. py:module:: lmflow.utils.data_utils

.. autoapi-nested-parse::

   The program includes several functions: setting a random seed, 
   loading data from a JSON file, batching data, and extracting answers from generated text.

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.data_utils.set_random_seed
   lmflow.utils.data_utils.load_data
   lmflow.utils.data_utils.batchlize
   lmflow.utils.data_utils.answer_extraction
   lmflow.utils.data_utils.process_image_flag



.. py:function:: set_random_seed(seed: int)

   
   Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.


   :Parameters:

       **seed** : int
           The default seed.














   ..
       !! processed by numpydoc !!

.. py:function:: load_data(file_name: str)

   
   Load data with file name.


   :Parameters:

       **file_name** : str.
           The dataset file name.

   :Returns:

       **inputs** : list.
           The input texts of the dataset.

       **outputs** : list.
           The output texts file datasets.    

       **len** : int.
           The length of the dataset.













   ..
       !! processed by numpydoc !!

.. py:function:: batchlize(examples: list, batch_size: int, random_shuffle: bool)

   
   Convert examples to a dataloader.


   :Parameters:

       **examples** : list.
           Data list.

       **batch_size** : int.
           ..

       **random_shuffle** : bool
           If true, the dataloader shuffle the training data.

   :Returns:

       dataloader:
           Dataloader with batch generator.













   ..
       !! processed by numpydoc !!

.. py:function:: answer_extraction(response, answer_type=None)

   
   Use this funtion to extract answers from generated text


   :Parameters:

       **args :**
           Arguments.

       **response** : str
           plain string response.

   :Returns:

       answer:
           Decoded answer (such as A, B, C, D, E for mutiple-choice QA).













   ..
       !! processed by numpydoc !!

.. py:function:: process_image_flag(text, image_flag='<ImageHere>')



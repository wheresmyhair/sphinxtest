:py:mod:`lmflow.datasets`
=========================

.. py:module:: lmflow.datasets

.. autoapi-nested-parse::

   
   This Python code defines a class Dataset with methods for initializing, loading,
   and manipulating datasets from different backends such as Hugging Face and JSON.

   The `Dataset` class includes methods for loading datasets from a dictionary and a Hugging
   Face dataset, mapping datasets, and retrieving the backend dataset and arguments.















   ..
       !! processed by numpydoc !!


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dataset/index.rst
   multi_modal_dataset/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.datasets.Dataset
   lmflow.datasets.CustomMultiModalDataset




.. py:class:: Dataset(data_args=None, backend: str = 'huggingface', *args, **kwargs)

   
   Initializes the Dataset object with the given parameters.


   :Parameters:

       **data_args** : DatasetArguments object.
           Contains the arguments required to load the dataset.

       **backend** : str,  default="huggingface"
           A string representing the dataset backend. Defaults to "huggingface".

       **args** : Optional.
           Positional arguments.

       **kwargs** : Optional.
           Keyword arguments.














   ..
       !! processed by numpydoc !!
   .. py:method:: __len__()


   .. py:method:: _check_data_format()

      
      Checks if data type and data structure matches

      Raise messages with hints if not matched.















      ..
          !! processed by numpydoc !!

   .. py:method:: from_dict(dict_obj: dict, *args, **kwargs)

      
      Create a Dataset object from a dictionary.

      Return a Dataset given a dict with format:
          {
              "type": TYPE,
              "instances": [
                  {
                      "key_1": VALUE_1.1,
                      "key_2": VALUE_1.2,
                      ...
                  },
                  {
                      "key_1": VALUE_2.1,
                      "key_2": VALUE_2.2,
                      ...
                  },
                  ...
              ]
          }

      :Parameters:

          **dict_obj** : dict.
              A dictionary containing the dataset information.

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          **self** : Dataset object.
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: create_from_dict(dict_obj, *args, **kwargs)
      :classmethod:

      




      :Returns:

          Returns a Dataset object given a dict.
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: to_dict()

      




      :Returns:

          Return a dict represents the dataset:
              {
                  "type": TYPE,
                  "instances": [
                      {
                          "key_1": VALUE_1.1,
                          "key_2": VALUE_1.2,
                          ...
                      },
                      {
                          "key_1": VALUE_2.1,
                          "key_2": VALUE_2.2,
                          ...
                      },
                      ...
                  ]
              }

          A python dict object represents the content of this dataset.
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: to_list()

      
      Returns a list of instances.
















      ..
          !! processed by numpydoc !!

   .. py:method:: map(*args, **kwargs)

      



      :Parameters:

          **args** : Optional.
              Positional arguments.

          **kwargs** : Optional.
              Keyword arguments.

      :Returns:

          **self** : Dataset object.
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: get_backend() -> Optional[str]

      




      :Returns:

          self.backend
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: get_backend_dataset()

      




      :Returns:

          self.backend_dataset
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: get_fingerprint()

      




      :Returns:

          Fingerprint of the backend_dataset which controls the cache
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: get_data_args()

      




      :Returns:

          self.data_args
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: get_type()

      




      :Returns:

          self.type
              ..













      ..
          !! processed by numpydoc !!


.. py:class:: CustomMultiModalDataset(dataset_path: str, data_args: lmflow.args.DatasetArguments)

   Bases: :py:obj:`torch.utils.data.Dataset`

   
   Dataset for Multi Modal data
















   ..
       !! processed by numpydoc !!
   .. py:method:: __len__()


   .. py:method:: register_tokenizer(tokenizer, image_processor=None)


   .. py:method:: __getitem__(i)




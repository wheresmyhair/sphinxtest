:py:mod:`lmflow.args`
=====================

.. py:module:: lmflow.args

.. autoapi-nested-parse::

   This script defines dataclasses: ModelArguments and DatasetArguments,
   that contain the arguments for the model and dataset used in training.

   It imports several modules, including dataclasses, field from typing, Optional from typing,
   require_version from transformers.utils.versions, MODEL_FOR_CAUSAL_LM_MAPPING,
   and TrainingArguments from transformers.

   MODEL_CONFIG_CLASSES is assigned a list of the model config classes from
   MODEL_FOR_CAUSAL_LM_MAPPING. MODEL_TYPES is assigned a tuple of the model types
   extracted from the MODEL_CONFIG_CLASSES.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.args.ModelArguments
   lmflow.args.VisModelArguments
   lmflow.args.DatasetArguments
   lmflow.args.MultiModalDatasetArguments
   lmflow.args.FinetunerArguments
   lmflow.args.EvaluatorArguments
   lmflow.args.InferencerArguments
   lmflow.args.RaftAlignerArguments
   lmflow.args.BenchmarkingArguments
   lmflow.args.AutoArguments




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.args.MODEL_CONFIG_CLASSES
   lmflow.args.MODEL_TYPES
   lmflow.args.PIPELINE_ARGUMENT_MAPPING


.. py:data:: MODEL_CONFIG_CLASSES
   

   

.. py:data:: MODEL_TYPES
   

   

.. py:class:: ModelArguments

   
   Define a class ModelArguments using the dataclass decorator. 
   The class contains several optional parameters that can be used to configure a model. 

   model_name_or_path : str
       a string representing the path or name of a pretrained
       model checkpoint for weights initialization. If None, a model will be trained from scratch.

   model_type :  str
       a string representing the type of model to use if training from
       scratch. If not provided, a pretrained model will be used.

   config_overrides :  str
       a string representing the default config settings to override
       when training a model from scratch.

   config_name : str
       a string representing the name or path of the pretrained config to
       use, if different from the model_name_or_path.

   tokenizer_name :  str
       a string representing the name or path of the pretrained tokenizer
       to use, if different from the model_name_or_path.

   cache_dir :  str
       a string representing the path to the directory where pretrained models
       downloaded from huggingface.co will be stored.

   use_fast_tokenizer : bool
       a boolean indicating whether to use a fast tokenizer (backed by the
       tokenizers library) or not.

   model_revision :  str
       a string representing the specific model version to use (can be a
       branch name, tag name, or commit id).

   use_auth_token : bool
       a boolean indicating whether to use the token generated when running
       huggingface-cli login (necessary to use this script with private models).

   torch_dtype :  str
       a string representing the dtype to load the model under. If auto is
       passed, the dtype will be automatically derived from the model's weights.

   use_ram_optimized_load : bool
       a boolean indicating whether to use disk mapping when memory is not
       enough.
   use_int8 : bool
       a boolean indicating whether to load int8 quantization for inference.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: model_name_or_path
      :annotation: :Optional[str]

      

   .. py:attribute:: lora_model_path
      :annotation: :Optional[str]

      

   .. py:attribute:: model_type
      :annotation: :Optional[str]

      

   .. py:attribute:: arch_type
      :annotation: :Optional[str]

      

   .. py:attribute:: config_overrides
      :annotation: :Optional[str]

      

   .. py:attribute:: arch_type
      :annotation: :Optional[str]

      

   .. py:attribute:: config_name
      :annotation: :Optional[str]

      

   .. py:attribute:: tokenizer_name
      :annotation: :Optional[str]

      

   .. py:attribute:: cache_dir
      :annotation: :Optional[str]

      

   .. py:attribute:: use_fast_tokenizer
      :annotation: :bool

      

   .. py:attribute:: model_revision
      :annotation: :str

      

   .. py:attribute:: use_auth_token
      :annotation: :bool

      

   .. py:attribute:: trust_remote_code
      :annotation: :bool

      

   .. py:attribute:: torch_dtype
      :annotation: :Optional[str]

      

   .. py:attribute:: use_lora
      :annotation: :bool

      

   .. py:attribute:: use_qlora
      :annotation: :bool

      

   .. py:attribute:: bits
      :annotation: :int

      

   .. py:attribute:: quant_type
      :annotation: :str

      

   .. py:attribute:: double_quant
      :annotation: :bool

      

   .. py:attribute:: lora_r
      :annotation: :int

      

   .. py:attribute:: lora_alpha
      :annotation: :int

      

   .. py:attribute:: lora_target_modules
      :annotation: :List[str]

      

   .. py:attribute:: lora_dropout
      :annotation: :float

      

   .. py:attribute:: save_aggregated_lora
      :annotation: :bool

      

   .. py:attribute:: use_ram_optimized_load
      :annotation: :bool

      

   .. py:attribute:: use_flash_attention
      :annotation: :bool

      

   .. py:attribute:: truncate_to_model_max_length
      :annotation: :bool

      

   .. py:attribute:: do_rope_scaling
      :annotation: :bool

      

   .. py:attribute:: rope_pi_ratio
      :annotation: :int

      

   .. py:attribute:: rope_ntk_ratio
      :annotation: :int

      

   .. py:attribute:: use_int8
      :annotation: :bool

      

   .. py:method:: __post_init__()



.. py:class:: VisModelArguments

   Bases: :py:obj:`ModelArguments`

   
   Define a class ModelArguments using the dataclass decorator. 
   The class contains several optional parameters that can be used to configure a model. 

   model_name_or_path : str
       a string representing the path or name of a pretrained
       model checkpoint for weights initialization. If None, a model will be trained from scratch.

   model_type :  str
       a string representing the type of model to use if training from
       scratch. If not provided, a pretrained model will be used.

   config_overrides :  str
       a string representing the default config settings to override
       when training a model from scratch.

   config_name : str
       a string representing the name or path of the pretrained config to
       use, if different from the model_name_or_path.

   tokenizer_name :  str
       a string representing the name or path of the pretrained tokenizer
       to use, if different from the model_name_or_path.

   cache_dir :  str
       a string representing the path to the directory where pretrained models
       downloaded from huggingface.co will be stored.

   use_fast_tokenizer : bool
       a boolean indicating whether to use a fast tokenizer (backed by the
       tokenizers library) or not.

   model_revision :  str
       a string representing the specific model version to use (can be a
       branch name, tag name, or commit id).

   use_auth_token : bool
       a boolean indicating whether to use the token generated when running
       huggingface-cli login (necessary to use this script with private models).

   torch_dtype :  str
       a string representing the dtype to load the model under. If auto is
       passed, the dtype will be automatically derived from the model's weights.

   use_ram_optimized_load : bool
       a boolean indicating whether to use disk mapping when memory is not
       enough.
   use_int8 : bool
       a boolean indicating whether to load int8 quantization for inference.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: low_resource
      :annotation: :Optional[bool]

      

   .. py:attribute:: custom_model
      :annotation: :bool

      

   .. py:attribute:: pretrained_language_projection_path
      :annotation: :str

      

   .. py:attribute:: custom_vision_model
      :annotation: :bool

      

   .. py:attribute:: image_encoder_name_or_path
      :annotation: :Optional[str]

      

   .. py:attribute:: qformer_name_or_path
      :annotation: :Optional[str]

      

   .. py:attribute:: llm_model_name_or_path
      :annotation: :Optional[str]

      

   .. py:attribute:: use_prompt_cache
      :annotation: :bool

      

   .. py:attribute:: prompt_cache_path
      :annotation: :Optional[str]

      

   .. py:attribute:: llava_loading
      :annotation: :Optional[bool]

      

   .. py:attribute:: with_qformer
      :annotation: :Optional[bool]

      

   .. py:attribute:: vision_select_layer
      :annotation: :Optional[int]

      

   .. py:attribute:: llava_pretrain_model_path
      :annotation: :Optional[str]

      

   .. py:attribute:: save_pretrain_model_path
      :annotation: :Optional[str]

      


.. py:class:: DatasetArguments

   
   Define a class DatasetArguments using the dataclass decorator. 
   The class contains several optional parameters that can be used to configure a dataset for a language model. 

   dataset_path : str
       a string representing the path of the dataset to use.

   dataset_name : str
       a string representing the name of the dataset to use. The default value is "customized".

   is_custom_dataset : bool
       a boolean indicating whether to use custom data. The default value is False.

   customized_cache_dir : str
       a string representing the path to the directory where customized dataset caches will be stored.

   dataset_config_name : str
       a string representing the configuration name of the dataset to use (via the datasets library).

   train_file : str
       a string representing the path to the input training data file (a text file).

   validation_file : str
       a string representing the path to the input evaluation data file to evaluate the perplexity on (a text file).

   max_train_samples : int
       an integer indicating the maximum number of training examples to use for debugging or quicker training. 
       If set, the training dataset will be truncated to this number.

   max_eval_samples: int
       an integer indicating the maximum number of evaluation examples to use for debugging or quicker training. 
       If set, the evaluation dataset will be truncated to this number.

   streaming : bool
       a boolean indicating whether to enable streaming mode.

   block_size: int
       an integer indicating the optional input sequence length after tokenization. The training dataset will be 
       truncated in blocks of this size for training.

   The class also includes some additional parameters that can be used to configure the dataset further, such as `overwrite_cache`,
   `validation_split_percentage`, `preprocessing_num_workers`, `disable_group_texts`, `demo_example_in_prompt`, `explanation_in_prompt`,
   `keep_linebreaks`, and `prompt_structure`.

   The field function is used to set default values and provide help messages for each parameter. The Optional type hint is
   used to indicate that a parameter is optional. The metadata argument is used to provide additional information about 
   each parameter, such as a help message.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: dataset_path
      :annotation: :Optional[str]

      

   .. py:attribute:: dataset_name
      :annotation: :Optional[str]

      

   .. py:attribute:: is_custom_dataset
      :annotation: :Optional[bool]

      

   .. py:attribute:: customized_cache_dir
      :annotation: :Optional[str]

      

   .. py:attribute:: dataset_config_name
      :annotation: :Optional[str]

      

   .. py:attribute:: train_file
      :annotation: :Optional[str]

      

   .. py:attribute:: validation_file
      :annotation: :Optional[str]

      

   .. py:attribute:: max_train_samples
      :annotation: :Optional[int]

      

   .. py:attribute:: max_eval_samples
      :annotation: :Optional[int]

      

   .. py:attribute:: streaming
      :annotation: :bool

      

   .. py:attribute:: block_size
      :annotation: :Optional[int]

      

   .. py:attribute:: overwrite_cache
      :annotation: :bool

      

   .. py:attribute:: validation_split_percentage
      :annotation: :Optional[int]

      

   .. py:attribute:: preprocessing_num_workers
      :annotation: :Optional[int]

      

   .. py:attribute:: group_texts_batch_size
      :annotation: :int

      

   .. py:attribute:: disable_group_texts
      :annotation: :bool

      

   .. py:attribute:: keep_linebreaks
      :annotation: :bool

      

   .. py:attribute:: test_file
      :annotation: :Optional[str]

      

   .. py:method:: __post_init__()



.. py:class:: MultiModalDatasetArguments

   Bases: :py:obj:`DatasetArguments`

   
   Define a class DatasetArguments using the dataclass decorator. 
   The class contains several optional parameters that can be used to configure a dataset for a language model. 

   dataset_path : str
       a string representing the path of the dataset to use.

   dataset_name : str
       a string representing the name of the dataset to use. The default value is "customized".

   is_custom_dataset : bool
       a boolean indicating whether to use custom data. The default value is False.

   customized_cache_dir : str
       a string representing the path to the directory where customized dataset caches will be stored.

   dataset_config_name : str
       a string representing the configuration name of the dataset to use (via the datasets library).

   train_file : str
       a string representing the path to the input training data file (a text file).

   validation_file : str
       a string representing the path to the input evaluation data file to evaluate the perplexity on (a text file).

   max_train_samples : int
       an integer indicating the maximum number of training examples to use for debugging or quicker training. 
       If set, the training dataset will be truncated to this number.

   max_eval_samples: int
       an integer indicating the maximum number of evaluation examples to use for debugging or quicker training. 
       If set, the evaluation dataset will be truncated to this number.

   streaming : bool
       a boolean indicating whether to enable streaming mode.

   block_size: int
       an integer indicating the optional input sequence length after tokenization. The training dataset will be 
       truncated in blocks of this size for training.

   The class also includes some additional parameters that can be used to configure the dataset further, such as `overwrite_cache`,
   `validation_split_percentage`, `preprocessing_num_workers`, `disable_group_texts`, `demo_example_in_prompt`, `explanation_in_prompt`,
   `keep_linebreaks`, and `prompt_structure`.

   The field function is used to set default values and provide help messages for each parameter. The Optional type hint is
   used to indicate that a parameter is optional. The metadata argument is used to provide additional information about 
   each parameter, such as a help message.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: image_folder
      :annotation: :Optional[str]

      

   .. py:attribute:: image_aspect_ratio
      :annotation: :Optional[str]

      

   .. py:attribute:: is_multimodal
      :annotation: :Optional[bool]

      

   .. py:attribute:: use_image_start_end
      :annotation: :Optional[bool]

      

   .. py:attribute:: sep_style
      :annotation: :Optional[str]

      


.. py:class:: FinetunerArguments

   Bases: :py:obj:`transformers.TrainingArguments`

   
   Adapt transformers.TrainingArguments
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: eval_dataset_path
      :annotation: :Optional[str]

      

   .. py:attribute:: remove_unused_columns
      :annotation: :Optional[bool]

      

   .. py:attribute:: finetune_part
      :annotation: :Optional[str]

      

   .. py:attribute:: save_language_projection
      :annotation: :Optional[str]

      


.. py:class:: EvaluatorArguments

   
   Define a class EvaluatorArguments using the dataclass decorator. The class contains several optional
   parameters that can be used to configure a evaluator.

   local_rank : str
       For distributed training: local_rank

   random_shuffle : bool

   use_wandb : bool

   random_seed : int, default = 1

   output_dir : str, default = './output_dir',

   mixed_precision : str, choice from ["bf16","fp16"].
       mixed precision mode, whether to use bf16 or fp16

   deepspeed : 
       Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
       loaded json file as a dict

   temperature : float
       An argument of model.generate in huggingface to control the diversity of generation.

   repetition_penalty : float
       An argument of model.generate in huggingface to penalize repetitions.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: local_rank
      :annotation: :int

      

   .. py:attribute:: random_shuffle
      :annotation: :Optional[bool]

      

   .. py:attribute:: use_wandb
      :annotation: :Optional[bool]

      

   .. py:attribute:: random_seed
      :annotation: :Optional[int]

      

   .. py:attribute:: output_dir
      :annotation: :Optional[str]

      

   .. py:attribute:: mixed_precision
      :annotation: :Optional[str]

      

   .. py:attribute:: deepspeed
      :annotation: :Optional[str]

      

   .. py:attribute:: answer_type
      :annotation: :Optional[str]

      

   .. py:attribute:: prompt_structure
      :annotation: :Optional[str]

      

   .. py:attribute:: evaluate_block_size
      :annotation: :Optional[int]

      

   .. py:attribute:: metric
      :annotation: :Optional[str]

      

   .. py:attribute:: inference_batch_size_per_device
      :annotation: :Optional[int]

      

   .. py:attribute:: use_accelerator_for_evaluator
      :annotation: :bool

      

   .. py:attribute:: temperature
      :annotation: :float

      

   .. py:attribute:: repetition_penalty
      :annotation: :float

      

   .. py:attribute:: max_new_tokens
      :annotation: :int

      


.. py:class:: InferencerArguments

   
   Define a class InferencerArguments using the dataclass decorator. The class contains several optional
   parameters that can be used to configure a inferencer.

   local_rank : str
       For distributed training: local_rank

   random_seed : int, default = 1

   deepspeed :
       Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
       loaded json file as a dict
   mixed_precision : str, choice from ["bf16","fp16"].
       mixed precision mode, whether to use bf16 or fp16

   temperature : float
       An argument of model.generate in huggingface to control the diversity of generation.

   repetition_penalty : float
       An argument of model.generate in huggingface to penalize repetitions.















   ..
       !! processed by numpydoc !!
   .. py:attribute:: device
      :annotation: :str

      

   .. py:attribute:: local_rank
      :annotation: :int

      

   .. py:attribute:: temperature
      :annotation: :float

      

   .. py:attribute:: repetition_penalty
      :annotation: :float

      

   .. py:attribute:: max_new_tokens
      :annotation: :int

      

   .. py:attribute:: random_seed
      :annotation: :Optional[int]

      

   .. py:attribute:: deepspeed
      :annotation: :Optional[str]

      

   .. py:attribute:: mixed_precision
      :annotation: :Optional[str]

      

   .. py:attribute:: do_sample
      :annotation: :Optional[bool]

      


.. py:class:: RaftAlignerArguments

   Bases: :py:obj:`transformers.TrainingArguments`

   
   Define a class RaftAlignerArguments to configure raft aligner.
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: output_reward_path
      :annotation: :Optional[str]

      

   .. py:attribute:: output_min_length
      :annotation: :Optional[int]

      

   .. py:attribute:: output_max_length
      :annotation: :Optional[int]

      

   .. py:attribute:: num_raft_iteration
      :annotation: :Optional[int]

      

   .. py:attribute:: raft_batch_size
      :annotation: :Optional[int]

      

   .. py:attribute:: top_reward_percentage
      :annotation: :Optional[float]

      

   .. py:attribute:: inference_batch_size_per_device
      :annotation: :Optional[int]

      

   .. py:attribute:: collection_strategy
      :annotation: :Optional[str]

      


.. py:class:: BenchmarkingArguments

   .. py:attribute:: dataset_name
      :annotation: :Optional[str]

      

   .. py:attribute:: lm_evaluation_metric
      :annotation: :Optional[str]

      


.. py:data:: PIPELINE_ARGUMENT_MAPPING
   

   

.. py:class:: AutoArguments

   
   Automatically choose arguments from FinetunerArguments or EvaluatorArguments.
















   ..
       !! processed by numpydoc !!
   .. py:method:: get_pipeline_args_class()




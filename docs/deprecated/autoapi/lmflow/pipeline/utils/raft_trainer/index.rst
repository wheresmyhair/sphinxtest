:py:mod:`lmflow.pipeline.utils.raft_trainer`
============================================

.. py:module:: lmflow.pipeline.utils.raft_trainer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.utils.raft_trainer.RaftTrainer




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.pipeline.utils.raft_trainer.is_torch_greater_or_equal_than_1_10
   lmflow.pipeline.utils.raft_trainer.is_torch_less_than_1_11
   lmflow.pipeline.utils.raft_trainer._is_native_cpu_amp_available
   lmflow.pipeline.utils.raft_trainer.DEFAULT_CALLBACKS
   lmflow.pipeline.utils.raft_trainer.DEFAULT_PROGRESS_CALLBACK
   lmflow.pipeline.utils.raft_trainer.DEFAULT_PROGRESS_CALLBACK
   lmflow.pipeline.utils.raft_trainer.IS_SAGEMAKER_MP_POST_1_10
   lmflow.pipeline.utils.raft_trainer.skip_first_batches
   lmflow.pipeline.utils.raft_trainer.logger
   lmflow.pipeline.utils.raft_trainer.TRAINING_ARGS_NAME
   lmflow.pipeline.utils.raft_trainer.TRAINER_STATE_NAME
   lmflow.pipeline.utils.raft_trainer.OPTIMIZER_NAME
   lmflow.pipeline.utils.raft_trainer.SCHEDULER_NAME
   lmflow.pipeline.utils.raft_trainer.SCALER_NAME


.. py:data:: is_torch_greater_or_equal_than_1_10
   

   

.. py:data:: is_torch_less_than_1_11
   

   

.. py:data:: _is_native_cpu_amp_available
   

   

.. py:data:: DEFAULT_CALLBACKS
   

   

.. py:data:: DEFAULT_PROGRESS_CALLBACK
   

   

.. py:data:: DEFAULT_PROGRESS_CALLBACK
   

   

.. py:data:: IS_SAGEMAKER_MP_POST_1_10
   

   

.. py:data:: skip_first_batches
   

   

.. py:data:: logger
   

   

.. py:data:: TRAINING_ARGS_NAME
   :annotation: = training_args.bin

   

.. py:data:: TRAINER_STATE_NAME
   :annotation: = trainer_state.json

   

.. py:data:: OPTIMIZER_NAME
   :annotation: = optimizer.pt

   

.. py:data:: SCHEDULER_NAME
   :annotation: = scheduler.pt

   

.. py:data:: SCALER_NAME
   :annotation: = scaler.pt

   

.. py:class:: RaftTrainer(model: Union[transformers.modeling_utils.PreTrainedModel, torch.nn.Module] = None, args: transformers.training_args.TrainingArguments = None, data_collator: Optional[transformers.data.data_collator.DataCollator] = None, train_dataset: Optional[torch.utils.data.Dataset] = None, eval_dataset: Optional[Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]] = None, tokenizer: Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None, model_init: Optional[Callable[[], transformers.modeling_utils.PreTrainedModel]] = None, compute_metrics: Optional[Callable[[transformers.trainer_utils.EvalPrediction], Dict]] = None, callbacks: Optional[List[transformers.trainer_callback.TrainerCallback]] = None, optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None)

   
   Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
   Args:
       model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
           The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
           <Tip>
           [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
           your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
           models.
           </Tip>
       args ([`TrainingArguments`], *optional*):
           The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
           `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
       data_collator (`DataCollator`, *optional*):
           The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
           default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
           [`DataCollatorWithPadding`] otherwise.
       train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`, *optional*):
           The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
           `model.forward()` method are automatically removed.
           Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
           distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
           `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
           manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
           sets the seed of the RNGs used.
       eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`]), *optional*):
            The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
            dataset prepending the dictionary key to the metric name.
       tokenizer ([`PreTrainedTokenizerBase`], *optional*):
           The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
           maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
           interrupted training or reuse the fine-tuned model.
       model_init (`Callable[[], PreTrainedModel]`, *optional*):
           A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
           from a new instance of the model as given by this function.
           The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
           be able to choose different architectures according to hyper parameters (such as layer count, sizes of
           inner layers, dropout probabilities etc).
       compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
           The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
           a dictionary string to metric values.
       callbacks (List of [`TrainerCallback`], *optional*):
           A list of callbacks to customize the training loop. Will add those to the list of default callbacks
           detailed in [here](callback).
           If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
       optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple
           containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
           and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
       preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
           A function that preprocess the logits right before caching them at each evaluation step. Must take two
           tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
           by this function will be reflected in the predictions received by `compute_metrics`.
           Note that the labels (second parameter) will be `None` if the dataset does not have them.
   Important attributes:
       - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
         subclass.
       - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
         original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
         the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
         model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
       - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
         data parallelism, this means some of the model layers are split on different GPUs).
       - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
         to `False` if model parallel or deepspeed is used, or if the default
         `TrainingArguments.place_model_on_device` is overridden to return `False` .
       - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
         in `train`)
















   ..
       !! processed by numpydoc !!
   .. py:method:: add_callback(callback)

      
      Add a callback to the current list of [`~transformer.TrainerCallback`].
      Args:
         callback (`type` or [`~transformer.TrainerCallback`]):
             A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
             first case, will instantiate a member of that class.
















      ..
          !! processed by numpydoc !!

   .. py:method:: pop_callback(callback)

      
      Remove a callback from the current list of [`~transformer.TrainerCallback`] and returns it.
      If the callback is not found, returns `None` (and no error is raised).
      Args:
         callback (`type` or [`~transformer.TrainerCallback`]):
             A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
             first case, will pop the first member of that class found in the list of callbacks.
      Returns:
          [`~transformer.TrainerCallback`]: The callback removed, if found.
















      ..
          !! processed by numpydoc !!

   .. py:method:: remove_callback(callback)

      
      Remove a callback from the current list of [`~transformer.TrainerCallback`].
      Args:
         callback (`type` or [`~transformer.TrainerCallback`]):
             A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
             first case, will remove the first member of that class found in the list of callbacks.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _move_model_to_device(model, device)


   .. py:method:: _set_signature_columns_if_needed()


   .. py:method:: _remove_unused_columns(dataset: datasets.Dataset, description: Optional[str] = None)


   .. py:method:: _get_collator_with_removed_columns(data_collator: Callable, description: Optional[str] = None) -> Callable

      
      Wrap the data collator in a callable removing unused columns.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _get_train_sampler() -> Optional[torch.utils.data.Sampler]


   .. py:method:: get_train_dataloader() -> torch.utils.data.DataLoader

      
      Returns the training [`~torch.utils.data.DataLoader`].
      Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
      training if necessary) otherwise.
      Subclass and override this method if you want to inject some custom behavior.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _get_eval_sampler(eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]


   .. py:method:: get_eval_dataloader(eval_dataset: Optional[torch.utils.data.Dataset] = None) -> torch.utils.data.DataLoader

      
      Returns the evaluation [`~torch.utils.data.DataLoader`].
      Subclass and override this method if you want to inject some custom behavior.
      Args:
          eval_dataset (`torch.utils.data.Dataset`, *optional*):
              If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
              by the `model.forward()` method are automatically removed. It must implement `__len__`.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_test_dataloader(test_dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader

      
      Returns the test [`~torch.utils.data.DataLoader`].
      Subclass and override this method if you want to inject some custom behavior.
      Args:
          test_dataset (`torch.utils.data.Dataset`, *optional*):
              The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
              `model.forward()` method are automatically removed. It must implement `__len__`.
















      ..
          !! processed by numpydoc !!

   .. py:method:: create_optimizer_and_scheduler(num_training_steps: int)

      
      Setup the optimizer and the learning rate scheduler.
      We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
      Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
      `create_scheduler`) in a subclass.
















      ..
          !! processed by numpydoc !!

   .. py:method:: create_optimizer()

      
      Setup the optimizer.
      We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
      Trainer's init through `optimizers`, or subclass and override this method in a subclass.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_optimizer_cls_and_kwargs(args: transformers.training_args.TrainingArguments) -> Tuple[Any, Any]
      :staticmethod:

      
      Returns the optimizer class and optimizer parameters based on the training arguments.
      Args:
          args (`transformers.training_args.TrainingArguments`):
              The training arguments for the training session.
















      ..
          !! processed by numpydoc !!

   .. py:method:: create_scheduler(num_training_steps: int, optimizer: torch.optim.Optimizer = None)

      
      Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
      passed as an argument.
      Args:
          num_training_steps (int): The number of training steps to do.
















      ..
          !! processed by numpydoc !!

   .. py:method:: num_examples(dataloader: torch.utils.data.DataLoader) -> int

      
      Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
      dataloader.dataset does not exist or has no length, estimates as best it can
















      ..
          !! processed by numpydoc !!

   .. py:method:: _hp_search_setup(trial: Union[optuna.Trial, Dict[str, Any]])

      
      HP search setup code
















      ..
          !! processed by numpydoc !!

   .. py:method:: _report_to_hp_search(trial: Union[optuna.Trial, Dict[str, Any]], step: int, metrics: Dict[str, float])


   .. py:method:: _tune_save_checkpoint()


   .. py:method:: call_model_init(trial=None)


   .. py:method:: torch_jit_model_eval(model, dataloader, training=False)


   .. py:method:: ipex_optimize_model(model, training=False, dtype=torch.float32)


   .. py:method:: _wrap_model(model, training=True, dataloader=None)


   .. py:method:: train(resume_from_checkpoint: Optional[Union[str, bool]] = None, trial: Union[optuna.Trial, Dict[str, Any]] = None, ignore_keys_for_eval: Optional[List[str]] = None, is_first_time=False, **kwargs)

      
      Main training entry point.
      Args:
          resume_from_checkpoint (`str` or `bool`, *optional*):
              If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
              `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
              of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
          trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
              The trial run or the hyperparameter dictionary for hyperparameter search.
          ignore_keys_for_eval (`List[str]`, *optional*)
              A list of keys in the output of your model (if it is a dictionary) that should be ignored when
              gathering predictions for evaluation during the training.
          kwargs:
              Additional keyword arguments used to hide deprecated arguments
















      ..
          !! processed by numpydoc !!

   .. py:method:: _one_train(batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None)


   .. py:method:: _inner_training_loop(batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None)

      
      0 This function serves to train one time
      1 Update the self.train_dataset before calling this function
















      ..
          !! processed by numpydoc !!

   .. py:method:: _get_output_dir(trial)


   .. py:method:: _load_from_checkpoint(resume_from_checkpoint, model=None)


   .. py:method:: _load_best_model()


   .. py:method:: _issue_warnings_after_load(load_result)


   .. py:method:: _maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


   .. py:method:: _load_rng_state(checkpoint)


   .. py:method:: _save_checkpoint(model, trial, metrics=None)


   .. py:method:: _load_optimizer_and_scheduler(checkpoint)

      
      If optimizer and scheduler states exist, load them.
















      ..
          !! processed by numpydoc !!

   .. py:method:: hyperparameter_search(hp_space: Optional[Callable[[optuna.Trial], Dict[str, float]]] = None, compute_objective: Optional[Callable[[Dict[str, float]], float]] = None, n_trials: int = 20, direction: str = 'minimize', backend: Optional[Union[str, transformers.trainer_utils.HPSearchBackend]] = None, hp_name: Optional[Callable[[optuna.Trial], str]] = None, **kwargs) -> transformers.trainer_utils.BestRun

      
      Launch an hyperparameter search using `optuna` or `Ray Tune` or `SigOpt`. The optimized quantity is determined
      by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
      the sum of all metrics otherwise.
      <Tip warning={true}>
      To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
      reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to
      subclass [`Trainer`] and override the method [`~Trainer.create_optimizer_and_scheduler`] for custom
      optimizer/scheduler.
      </Tip>
      Args:
          hp_space (`Callable[["optuna.Trial"], Dict[str, float]]`, *optional*):
              A function that defines the hyperparameter search space. Will default to
              [`~trainer_utils.default_hp_space_optuna`] or [`~trainer_utils.default_hp_space_ray`] or
              [`~trainer_utils.default_hp_space_sigopt`] depending on your backend.
          compute_objective (`Callable[[Dict[str, float]], float]`, *optional*):
              A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
              method. Will default to [`~trainer_utils.default_compute_objective`].
          n_trials (`int`, *optional*, defaults to 100):
              The number of trial runs to test.
          direction (`str`, *optional*, defaults to `"minimize"`):
              Whether to optimize greater or lower objects. Can be `"minimize"` or `"maximize"`, you should pick
              `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or several metrics.
          backend (`str` or [`~training_utils.HPSearchBackend`], *optional*):
              The backend to use for hyperparameter search. Will default to optuna or Ray Tune or SigOpt, depending
              on which one is installed. If all are installed, will default to optuna.
          hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
              A function that defines the trial/run name. Will default to None.
          kwargs (`Dict[str, Any]`, *optional*):
              Additional keyword arguments passed along to `optuna.create_study` or `ray.tune.run`. For more
              information see:
              - the documentation of
                [optuna.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
              - the documentation of [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run)
              - the documentation of [sigopt](https://app.sigopt.com/docs/endpoints/experiments/create)
      Returns:
          [`trainer_utils.BestRun`]: All the information about the best run. Experiment summary can be found in
          `run_summary` attribute for Ray backend.
















      ..
          !! processed by numpydoc !!

   .. py:method:: log(logs: Dict[str, float]) -> None

      
      Log `logs` on the various objects watching training.
      Subclass and override this method to inject custom behavior.
      Args:
          logs (`Dict[str, float]`):
              The values to log.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]

      
      Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]

      
      Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
      handling potential state.
















      ..
          !! processed by numpydoc !!

   .. py:method:: compute_loss_context_manager()

      
      A helper wrapper to group together context managers.
















      ..
          !! processed by numpydoc !!

   .. py:method:: autocast_smart_context_manager(cache_enabled: Optional[bool] = True)

      
      A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
      arguments, depending on the situation.
















      ..
          !! processed by numpydoc !!

   .. py:method:: training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor

      
      Perform a training step on a batch of inputs.
      Subclass and override to inject custom behavior.
      Args:
          model (`nn.Module`):
              The model to train.
          inputs (`Dict[str, Union[torch.Tensor, Any]]`):
              The inputs and targets of the model.
              The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
              argument `labels`. Check your model's documentation for all accepted arguments.
      Return:
          `torch.Tensor`: The tensor with training loss on this batch.
















      ..
          !! processed by numpydoc !!

   .. py:method:: compute_loss(model, inputs, return_outputs=False)

      
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
















      ..
          !! processed by numpydoc !!

   .. py:method:: is_local_process_zero() -> bool

      
      Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
      machines) main process.
















      ..
          !! processed by numpydoc !!

   .. py:method:: is_world_process_zero() -> bool

      
      Whether or not this process is the global main process (when training in a distributed fashion on several
      machines, this is only going to be `True` for one process).
















      ..
          !! processed by numpydoc !!

   .. py:method:: save_model(output_dir: Optional[str] = None, _internal_call: bool = False)

      
      Will save the model, so you can reload it using `from_pretrained()`.
      Will only save from the main process.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _save_tpu(output_dir: Optional[str] = None)


   .. py:method:: _save(output_dir: Optional[str] = None, state_dict=None)


   .. py:method:: store_flos()


   .. py:method:: _sorted_checkpoints(output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]


   .. py:method:: _rotate_checkpoints(use_mtime=False, output_dir=None) -> None


   .. py:method:: evaluate(eval_dataset: Optional[torch.utils.data.Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = 'eval') -> Dict[str, float]

      
      Run evaluation and returns metrics.
      The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
      (pass it to the init `compute_metrics` argument).
      You can also subclass and override this method to inject custom behavior.
      Args:
          eval_dataset (`Dataset`, *optional*):
              Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
              not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
              method.
          ignore_keys (`Lst[str]`, *optional*):
              A list of keys in the output of your model (if it is a dictionary) that should be ignored when
              gathering predictions.
          metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
              An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
              "eval_bleu" if the prefix is "eval" (default)
      Returns:
          A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
          dictionary also contains the epoch number which comes from the training state.
















      ..
          !! processed by numpydoc !!

   .. py:method:: predict(test_dataset: torch.utils.data.Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = 'test') -> transformers.trainer_utils.PredictionOutput

      
      Run prediction and returns predictions and potential metrics.
      Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
      will also return metrics, like in `evaluate()`.
      Args:
          test_dataset (`Dataset`):
              Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
              `model.forward()` method are automatically removed. Has to implement the method `__len__`
          ignore_keys (`Lst[str]`, *optional*):
              A list of keys in the output of your model (if it is a dictionary) that should be ignored when
              gathering predictions.
          metric_key_prefix (`str`, *optional*, defaults to `"test"`):
              An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
              "test_bleu" if the prefix is "test" (default)
      <Tip>
      If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
      in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
      one array. The padding index is -100.
      </Tip>
      Returns: *NamedTuple* A namedtuple with the following keys:
          - predictions (`np.ndarray`): The predictions on `test_dataset`.
          - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
          - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
            labels).
















      ..
          !! processed by numpydoc !!

   .. py:method:: evaluation_loop(dataloader: torch.utils.data.DataLoader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = 'eval') -> transformers.trainer_utils.EvalLoopOutput

      
      Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
      Works both with or without labels.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _nested_gather(tensors, name=None)

      
      Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
      concatenating them to `gathered`
















      ..
          !! processed by numpydoc !!

   .. py:method:: _pad_across_processes(tensor, pad_index=-100)

      
      Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
      they can safely be gathered.
















      ..
          !! processed by numpydoc !!

   .. py:method:: prediction_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

      
      Perform an evaluation step on `model` using `inputs`.
      Subclass and override to inject custom behavior.
      Args:
          model (`nn.Module`):
              The model to evaluate.
          inputs (`Dict[str, Union[torch.Tensor, Any]]`):
              The inputs and targets of the model.
              The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
              argument `labels`. Check your model's documentation for all accepted arguments.
          prediction_loss_only (`bool`):
              Whether or not to return the loss only.
          ignore_keys (`Lst[str]`, *optional*):
              A list of keys in the output of your model (if it is a dictionary) that should be ignored when
              gathering predictions.
      Return:
          Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
          logits and labels (each being optional).
















      ..
          !! processed by numpydoc !!

   .. py:method:: floating_point_ops(inputs: Dict[str, Union[torch.Tensor, Any]])

      
      For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
      operations for every backward + forward pass. If using another model, either implement such a method in the
      model or subclass and override this method.
      Args:
          inputs (`Dict[str, Union[torch.Tensor, Any]]`):
              The inputs and targets of the model.
      Returns:
          `int`: The number of floating-point operations.
















      ..
          !! processed by numpydoc !!

   .. py:method:: init_git_repo(at_init: bool = False)

      
      Initializes a git repo in `self.args.hub_model_id`.
      Args:
          at_init (`bool`, *optional*, defaults to `False`):
              Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is
              `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped
              out.
















      ..
          !! processed by numpydoc !!

   .. py:method:: create_model_card(language: Optional[str] = None, license: Optional[str] = None, tags: Union[str, List[str], None] = None, model_name: Optional[str] = None, finetuned_from: Optional[str] = None, tasks: Union[str, List[str], None] = None, dataset_tags: Union[str, List[str], None] = None, dataset: Union[str, List[str], None] = None, dataset_args: Union[str, List[str], None] = None)

      
      Creates a draft of a model card using the information available to the `Trainer`.
      Args:
          language (`str`, *optional*):
              The language of the model (if applicable)
          license (`str`, *optional*):
              The license of the model. Will default to the license of the pretrained model used, if the original
              model given to the `Trainer` comes from a repo on the Hub.
          tags (`str` or `List[str]`, *optional*):
              Some tags to be included in the metadata of the model card.
          model_name (`str`, *optional*):
              The name of the model.
          finetuned_from (`str`, *optional*):
              The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
              of the original model given to the `Trainer` (if it comes from the Hub).
          tasks (`str` or `List[str]`, *optional*):
              One or several task identifiers, to be included in the metadata of the model card.
          dataset_tags (`str` or `List[str]`, *optional*):
              One or several dataset tags, to be included in the metadata of the model card.
          dataset (`str` or `List[str]`, *optional*):
              One or several dataset identifiers, to be included in the metadata of the model card.
          dataset_args (`str` or `List[str]`, *optional*):
             One or several dataset arguments, to be included in the metadata of the model card.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _push_from_checkpoint(checkpoint_folder)


   .. py:method:: push_to_hub(commit_message: Optional[str] = 'End of training', blocking: bool = True, **kwargs) -> str

      
      Upload *self.model* and *self.tokenizer* to the 🤗 model hub on the repo *self.args.hub_model_id*.
      Parameters:
          commit_message (`str`, *optional*, defaults to `"End of training"`):
              Message to commit while pushing.
          blocking (`bool`, *optional*, defaults to `True`):
              Whether the function should return only when the `git push` has finished.
          kwargs:
              Additional keyword arguments passed along to [`~Trainer.create_model_card`].
      Returns:
          The url of the commit of your model in the given repository if `blocking=False`, a tuple with the url of
          the commit and an object to track the progress of the commit if `blocking=True`
















      ..
          !! processed by numpydoc !!

   .. py:method:: prediction_loop(dataloader: torch.utils.data.DataLoader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = 'eval') -> transformers.trainer_utils.EvalLoopOutput

      
      Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
      Works both with or without labels.
















      ..
          !! processed by numpydoc !!

   .. py:method:: _gather_and_numpify(tensors, name)

      
      Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
      concatenating them to `gathered`
















      ..
          !! processed by numpydoc !!

   .. py:method:: _add_sm_patterns_to_gitignore() -> None

      
      Add SageMaker Checkpointing patterns to .gitignore file.
















      ..
          !! processed by numpydoc !!



import logging
import os
import pprint
import sys
import tempfile as tmp

from autokeras import StructuredDataClassifier, StructuredDataRegressor, __version__
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO and WARNING messages are not printed
import tensorflow as tf
from tensorflow.keras import metrics, losses
from keras import backend as K

from frameworks.shared.callee import call_run, output_subdir, result
from frameworks.shared.utils import Timer, is_sparse


log = logging.getLogger(__name__)

def _f1(y, pred):
    precision = metrics.Precision(y, pred)
    recall = metrics.Recall(y, pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def run(dataset, config):
    log.info(f"\n**** AutoKeras [v{__version__}]****\n")

    is_classification = config.type == 'classification'
    # Mapping of benchmark metrics to TPOT metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='AUC',
        f1=(lambda y, pred: _f1(y, pred), False),
        logloss='logloss',
        mae='MeanAbsoluteError',
        mse='MeanSquaredError',
        msle='MeanSquaredLogarithmicError',
        # r2='r2',
        rmse='RootMeanSquaredError',
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    # tfcfg = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=n_jobs,
    #                         inter_op_parallelism_threads=n_jobs, 
    #                         allow_soft_placement=True
    #                         )
    # session = tf.compat.v1.Session(config=tfcfg)
    # K.set_session(session) 
    output_dir = config.output_dir
    project_name = config.framework + config.name
    runtime_min = (config.max_runtime_seconds/60)
    log.info(f'Running AutoKeras on {n_jobs} cores, optimizing {scoring_metric}.')
    log.info(f'Please, note that the time limit of {runtime_min} cannot be enforced on Keras.')

    X_train = dataset.train.X
    y_train = dataset.train.y
    is_classification = config.type == 'classification'
    log.info("Running AutoKeras for {config.type}.")
    if  is_classification: 
        estimator = StructuredDataClassifier
        if dataset.problem_type is 'binary':
            lossfn = losses.BinaryCrossEntropy()
            objective = 'binary_crossentropy'  # logloss
        elif dataset.problem_type is 'multiclass':
            lossfn = losses.CategoricalCrossEntropy()
            objective = 'categorical_crossentropy'
    else:  # 'regression'
        estimator = StructuredDataRegressor
        lossfn = losses.MeanSquaredError()
        objective = 'mean_squared_error'
    #     loss: A Keras loss function. Defaults to use 'mean_squared_error'.
    #     metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
    #     project_name: String. The name of the AutoModel. Defaults to
    #         'structured_data_regressor'.
    #     max_trials: Int. The maximum number of different Keras Models to try.
    #         The search may finish before reaching the max_trials. Defaults to 100.
    #     directory: String. The path to a directory for storing the search outputs.
    #         Defaults to None, which would create a folder with the name of the
    #         AutoModel in the current directory.
    #     objective: String. Name of model metric to minimize
    #         or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
    #     tuner: String or subclass of AutoTuner. If string, it should be one of
    #         'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass
    #         of AutoTuner. If left unspecified, it uses a task specific tuner, which
    #         first evaluates the most commonly used models for the task before
    #         exploring other models.
    #     overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
    #         project of the same name if one is found. Otherwise, overwrites the
    #         project.
    #     seed: Int. Random seed.
    #     max_model_size: Int. Maximum number of scalars in the parameters of a
    #         model. Models larger than this are rejected.

    aks = estimator(project_name=project_name,
                    directory=output_dir,
                    #metrics=scoring_metric,
                    seed=config.seed,
                    #loss=lossfn,
                    #objective=objective,
                    **training_params)  # overwrite=True

    with Timer() as training:
        #     x: numpy.ndarray or tensorflow.Dataset. Training data x.
        #     y: numpy.ndarray or tensorflow.Dataset. Training data y.
        #     batch_size: Int. Number of samples per gradient update. Defaults to 32.
        #     epochs: Int. The number of epochs to train each model during the search.
        #         If unspecified, by default we train for a maximum of 1000 epochs,
        #         but we stop training if the validation loss stops improving for 10
        #         epochs (unless you specified an EarlyStopping callback as part of
        #         the callbacks argument, in which case the EarlyStopping callback you
        #         specified will determine early stopping).
        #     callbacks: List of Keras callbacks to apply during training and
        #         validation.
        #     validation_split: Float between 0 and 1. Defaults to 0.2.
        #         Fraction of the training data to be used as validation data.
        #         The model will set apart this fraction of the training data,
        #         will not train on it, and will evaluate
        #         the loss and any model metrics
        #         on this data at the end of each epoch.
        #         The validation data is selected from the last samples
        #         in the `x` and `y` data provided, before shuffling. This argument is
        #         not supported when `x` is a dataset.
        #         The best model found would be fit on the entire dataset including the
        #         validation data.
        #     validation_data: Data on which to evaluate the loss and any model metrics
        #         at the end of each epoch. The model will not be trained on this data.
        #         `validation_data` will override `validation_split`. The type of the
        #         validation data should be the same as the training data.
        #         The best model found would be fit on the training dataset without the
        #         validation data.
        #     verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
        #         2 = one line per epoch. Note that the progress bar is not
        #         particularly useful when logged to a file, so verbose=2 is
        #         recommended when not running interactively (eg, in a production
        #         environment). Controls the verbosity of both KerasTuner search and
        #         [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
        aks.fit(X_train, y_train, epochs=2, batch_size=16, verbose=1)

    log.info('Predicting on the test set.')
    X_test = dataset.test.X
    y_test = dataset.test.y
    with Timer() as predict:
        log.info('Predicting on the test set.')
        predictions = aks.predict(X_test)
    probabilities = aks.export_model().predict(X_test) if is_classification else None
 
    log.info('Saving the artifacts.')
    save_artifacts(aks, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  #models_count=len(aks.evaluated_individuals_),
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def save_artifacts(estimator, config):
    try:
        log.debug("All individuals :\n%s", list(estimator.evaluated_individuals_.items()))
        models = estimator.pareto_front_fitted_pipelines_
        hall_of_fame = list(zip(reversed(estimator._pareto_front.keys), estimator._pareto_front.items))
        artifacts = config.framework_params.get('_save_artifacts', False)
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                for m in hall_of_fame:
                    pprint.pprint(dict(
                        fitness=str(m[0]),
                        model=str(m[1]),
                        pipeline=models[str(m[1])],
                    ), stream=f)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)

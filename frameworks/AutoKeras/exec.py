import logging
import os
import pprint
import sys
import tempfile as tmp

from autokeras import StructuredDataClassifier, StructuredDataRegressor, __version__
from tensorflow.keras import metrics
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
        acc=metrics.Accuracy(),
        auc=metrics.AUC(),
        f1=(lambda y, pred: _f1(y, pred), False),
        # logloss='logloss',
        mae=metrics.MeanAbsoluteError(),
        mse=metrics.MeanSquaredError(),
        msle=metrics.MeanSquaredLogarithmicError(),
        # r2='r2',
        rmse=metrics.RootMeanSquaredError(),
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train = dataset.train.X
    y_train = dataset.train.y

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config
    config_dict = config.framework_params.get('_config_dict', "TPOT sparse" if is_sparse(X_train) else None)

    log.info('Running AutoKeras with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)
    runtime_min = (config.max_runtime_seconds/60)

(project_name=project, directory=destination, max_trials=trials, seed=1977,
                                       overwrite=True)
    estimator = StructuredDataClassifier if is_classification else StructuredDataRegressor
    tpot = estimator(n_jobs=n_jobs,
                     max_time_mins=runtime_min,
                     scoring=scoring_metric,
                     random_state=config.seed,
                     config_dict=config_dict,
                     **training_params)

    with Timer() as training:
        tpot.fit(X_train, y_train)

    log.info('Predicting on the test set.')
    X_test = dataset.test.X
    y_test = dataset.test.y
    with Timer() as predict:
        predictions = tpot.predict(X_test)
    try:
        probabilities = tpot.predict_proba(X_test) if is_classification else None
    except RuntimeError:
        # TPOT throws a RuntimeError if the optimized pipeline does not support `predict_proba`.
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`

    save_artifacts(tpot, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(tpot.evaluated_individuals_),
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
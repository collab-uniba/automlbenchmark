import logging
import os
import pprint

import numpy as np
import pandas as pd
from autoviml import __version__
from autoviml.Auto_ViML import Auto_ViML
from frameworks.shared.callee import call_run, output_subdir, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n****  AutoViML [v{__version__}]****\n")

    # Mapping of benchmark metrics to AutoViML metrics
    metrics_mapping = dict(
        acc="accuracy",
        auc="roc_auc",
        f1="f1",
        logloss="log_loss",
        mae="mean_absolute_error",
        mse="mean_squared_error",
        rmse="root_mean_squared_error",
        r2="r2",
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith("_")}
    n_jobs = config.framework_params.get(
        "_n_jobs", config.cores
    )  # useful to disable multicore, regardless of the dataset config
    n_jobs = min(n_jobs, config.cores)

    # runtime_min = config.max_runtime_seconds / 60
    # log.info(
    #     f"Running AutoViML on {n_jobs} cores, optimizing {scoring_metric} with a time limit of {runtime_min}."
    # )

    target = dataset.target
    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    y_test = dataset.test.y
    train, test = X_train.join(y_train), X_test.join(y_test)

    # You don't have to tell Auto_ViML whether it is a Regression or
    # Classification problem.
    is_classification = config.type == "classification"
    dataset_name = config.name

    log.info(f"Running AutoViML for {is_classification} on dataset {dataset_name}.")
    log.debug("Environment: %s", os.environ)
    predictors_type = dataset.predictors_type
    log.debug("predictors_type=%s", predictors_type)

    # Parameters for AutoViML
    # train: could be a datapath+filename or a dataframe. It will detect which
    #       is which and load it.
    # test: could be a datapath+filename or a dataframe. If you don't have any,
    #       just leave it as "".
    # submission: must be a datapath+filename. If you don't have any, just
    #       leave it as empty string.
    # target: name of the target variable in the data set.
    # sep: if you have a spearator in the file such as "," or "\t" mention it
    #       here. Default is ",".
    # scoring_parameter: if you want your own scoring parameter such as "f1"
    #       give it here. If not, it will assume the appropriate scoring param
    #       for the problem and it will build the model.
    # hyper_param: Tuning options are GridSearch ('GS') and RandomizedSearch
    #       ('RS'). Default is 'RS'.
    # feature_reduction: Default = 'True' but it can be set to False if you
    #       don't want automatic feature_reduction since in Image data sets
    #       like digits and MNIST, you get better results when you don't
    #       reduce features automatically. You can always try both and see.
    # KMeans_Featurizer
    #       True: Adds a cluster label to features based on KMeans.
    #             Use for Linear.
    #       False (default) For Random Forests or XGB models, leave it False
    #             since it may overfit.
    # Boosting Flag: you have 4 possible choices (default is False):
    #       None This will build a Linear Model
    #       False This will build a Random Forest or Extra Trees model (also
    #             known as Bagging)
    #       True This will build an XGBoost model
    # CatBoost This will build a CatBoost model (provided you have CatBoost
    #       installed)
    # Add_Poly: Default is 0 which means do-nothing. But it has three
    #       interesting settings:
    #           1 Add interaction variables only e.g. x1x2,x2x3,...x9*10 etc.
    #           2 Add Interactions and Squared variables e.g. x12,x22, etc.
    #           3 Adds both Interactions and Squared variables e.g. x1x2,x1**2,
    #             x2x3,x2**2, etc.
    # Stacking_Flag: Default is False. If set to True, it will add an
    #       additional feature which is derived from predictions of another
    #       model. This is used in some cases but may result in overfitting.
    #       So be careful turning this flag "on".
    # Binning_Flag: Default is False. It set to True, it will convert the top
    #       numeric variables into binned variables through a technique known
    #       as "Entropy" binning. This is very helpful for certain datasets
    #       (especially hard to build models).
    # Imbalanced_Flag: Default is False. If set to True, it will use SMOTE from
    #       Imbalanced-Learn to oversample the "Rare Class" in an imbalanced
    #       dataset and make the classes balanced (50-50 for example in a
    #       binary classification). This also works for Regression problems
    #       where you have highly skewed distributions in the target variable.
    #       Auto_ViML creates additional samples using SMOTE for Highly
    #       Imbalanced data.
    # verbose: This has 3 possible states:
    #       0 limited output. Great for running this silently and getting fast
    #         results.
    #       1 more charts. Great for knowing how results were and making
    #         changes to flags in input.
    #       2 lots of charts and output. Great for reproducing what Auto_ViML
    #         does on your own.
    #
    # Return values
    # model: It will return your trained model
    # features: the fewest number of features in your model to make it perform
    #       well
    # train_modified: this is the modified train dataframe after removing and
    #       adding features
    # test_modified: this is the modified test dataframe with the same
    #       transformations as train
    log.info("Training and predicting.")
    # Training and prediction stages here happen together.
    with Timer() as training:
        with Timer() as predict:
            # Note: this uses its own interna seed, non need to pass config.seed
            model, features, train_modified, test_modified = Auto_ViML(
                train=train,
                test=test,
                target=target,
                scoring_parameter=scoring_metric,
                **training_params,
            )

    labels = dataset.labels
    if is_classification:
        type = "Binary" if dataset.problem_type == "binary" else "Multi"
        output_predictions_file = os.path.join(
            target, f"{target}_{type}_Classification_submission.csv"
        )
        predictions = pd.read_csv(output_predictions_file, header=None, skiprows=1)
        # probabilities = np.zeros(len(predictions.index)).tolist()  # TODO
        # probabilities = np.ndarray(shape=(2,len(predictions.index)), dtype=float, buffer=np.zeros(len(predictions.index)))
        probabilities = pd.DataFrame(
            index=predictions.index,
            columns=labels,
            dtype=float,
        )
        p = 1 / len(labels)
        for label in labels:
            probabilities[label] = p
    else:
        probabilities = None
        output_predictions_file = os.path.join(target, f"{target}_Regression_submission.csv")
        predictions = pd.read_csv(output_predictions_file, header=None, skiprows=1)

    log.info("Saving the artifacts.")
    # save_artifacts(model, features, config)

    return result(
        output_file=config.output_predictions_file,
        # process_results=process_results,
        predictions=predictions,
        probabilities=probabilities,
        probabilities_labels=labels,
        target_is_encoded=False,
        # models_count=len(model),
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def process_results(res):

    return None


def save_artifacts(estimator, config):
    try:
        log.debug("All individuals :\n%s", list(estimator.evaluated_individuals_.items()))
        models = estimator.pareto_front_fitted_pipelines_
        hall_of_fame = list(
            zip(reversed(estimator._pareto_front.keys), estimator._pareto_front.items)
        )
        artifacts = config.framework_params.get("_save_artifacts", False)
        if "models" in artifacts:
            models_file = os.path.join(output_subdir("models", config), "models.txt")
            with open(models_file, "w") as f:
                for m in hall_of_fame:
                    pprint.pprint(
                        dict(
                            fitness=str(m[0]),
                            model=str(m[1]),
                            pipeline=models[str(m[1])],
                        ),
                        stream=f,
                    )
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == "__main__":
    call_run(run)

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir

import os

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X=dataset.train.X,
            y=dataset.train.y
        ),
        test=dict(
            X=dataset.test.X,
            y=dataset.test.y
        ),
        problem_type=dataset.type.name
    )

    os.environ['PATH'] = '/usr/lib/x86_64-linux-gnu${PATH:+:${PATH}}'
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
    print(f"**********PATH environment: {os.environ['PATH']}")
    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

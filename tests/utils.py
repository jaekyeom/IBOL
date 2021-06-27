import datetime
import os
import socket

from garage.experiment.experiment import get_metadata

import aim_wrapper

def get_run_env_dict():
    d = {}
    d['timestamp'] = datetime.datetime.now().timestamp()
    d['hostname'] = socket.gethostname()
    if 'SLURM_JOB_ID' in os.environ:
        d['slurm_job_id'] = int(os.environ['SLURM_JOB_ID'])
    if 'SLURM_PROCID' in os.environ:
        d['slurm_procid'] = int(os.environ['SLURM_PROCID'])
    if 'SLURM_RESTART_COUNT' in os.environ:
        d['slurm_restart_count'] = int(os.environ['SLURM_RESTART_COUNT'])

    git_root_path, metadata = get_metadata()
    # get_metadata() does not decode git_root_path.
    d['git_root_path'] = git_root_path.decode('utf-8') if git_root_path is not None else None
    d['git_commit'] = metadata.get('githash')
    d['launcher'] = metadata.get('launcher')

    return d

def construct_with_aim_logging(cls, kwargs, name, excluded_keys=[]):
    # Although unlikely, some module classes might update input args, thus we log first.
    aim_wrapper.set_params({'class': cls.__name__}, name=name)
    aim_wrapper.set_params(kwargs, name=name, excluded_keys=excluded_keys)
    module = cls(**kwargs)
    return module

"""
Simple façade so that `from docker_smpc_manager import …` works.
"""
from .docker_deployer import (
    build_image,
    fan_out_and_run,
    read_outputs,
    average_models,
    apply_global_model,
)
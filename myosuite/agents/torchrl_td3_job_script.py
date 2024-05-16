import hydra

import torch.nn
import torch.optim
import numpy as np

from torchrl.record.loggers import generate_exp_name, get_logger

@hydra.main(config_path="", config_name="config")
def main(cfg: "DictConfig"):
    device = cfg.network.device
    if device in ("", None):
        raise ValueError("Specify device in config file.")

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TD3-DEP", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="td3-dep_logger",
            experiment_name=exp_name
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create myosuite environment

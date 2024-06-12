import hydra

import torch.nn
import torch.optim
import numpy as np

from torchrl.record.loggers import generate_exp_name, get_logger

from torchrl_utils import *

@hydra.main(config_path="", config_name="config")
def main(cfg: "DictConfig"):
    device = cfg.network.device
    if device in ("", None):
        raise ValueError("Specify device in config file.")
    device = torch.device(device)

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TD3-DEP", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3_dep_logging",
            experiment_name=exp_name,
            wandb_kwargs={

            }
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create myosuite environment // starting with a single worker TODO: support more
    env = make_env(cfg.env.env_name, device=cfg.env.device)

    # Create agent
    actor, critic, exploration_policy = make_td3_models(cfg)

    # Create TD3 loss
    loss_module, target_net_update = make_loss_module(cfg, actor, critic)

    # Create off-policy collector
    collector = make_collector(cfg, env, exploration_policy)


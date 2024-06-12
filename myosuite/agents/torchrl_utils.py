from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)

from torchrl.modules import (
    AdditiveGaussianWrapper,
    MLP,
    SafeModule,
    SafeSequential,
    TanhModule,
    ValueOperator,
)

from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, SafeModule
from torchrl.collectors import SyncDataCollector
import torch
from torch import nn

from torchrl.objectives.td3 import TD3Loss
from torchrl.objectives import SoftUpdate

from myosuite.utils import gym
from torchrl.envs.utils import ExplorationType, set_exploration_type


def make_env(env_name="myoElbowPose1D6MRandom-v0", device="cpu"):
    env = GymWrapper(gym.make(env_name), device=device)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat())
    return env


def make_td3_models(cfg):
    """Make TD3 agent"""
    proof_environment = make_env(cfg.env.env_name)
    # Define actor network
    input_shape = proof_environment.observation_spec["observation"].shape
    num_outputs = proof_environment.action_spec.shape[-1]
    # could a batch of actions be beneficial?
    action_spec = proof_environment.action_spec

    actor_net = MLP(
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,
        num_cells=cfg.network.num_cells
    )

    # Initialize actor weights
    for layer in actor_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    actor_module = SafeModule(
        actor_net,
        in_keys=input_shape,
        out_keys=[
            "param"
        ],
    )

    actor = SafeSequential(
        actor_module,
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec
        ),
    )

    critic_net = MLP(
        num_cells=cfg.network.num_cells,
        out_features=1,
        activation_class=nn.Tanh
    )

    # Initialize critic weights
    for layer in critic_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    critic = ValueOperator(
        critic_net,
        in_keys=["observation"]
    )

    # init nets / unsure if this is actually needed
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = proof_environment.reset()
        td = td.to(cfg.network.device)
        actor(td)
        critic(td)
    del td
    proof_environment.close()

    # default exploration wrapper that adds gaussian noise to the actor's actions
    # TODO: replace with DEP
    exploration_wrapper = AdditiveGaussianWrapper(
        actor,
        sigma_init=1,
        sigma_end=1,
        mean=0,
        std=0.1,
        spec=action_spec,
    ).to(cfg.network.device)

    return actor, critic, exploration_wrapper


def make_loss_module(cfg, actor, critic):
    """Make loss module and target network updater"""
    # Create TD3 loss
    loss_module = TD3Loss(
        actor_network=actor,
        qvalue_network=critic,
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=True,
        delay_qvalue=True,
        action_spec=actor[1].spec,
        policy_noise=cfg.optim.policy_noise,
        noise_clip=cfg.optim.noise_clip,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define target network updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater

def make_collector(cfg, train_env, exploration_policy):
    """Make off-policy collector"""
    collector = SyncDataCollector(
        train_env,
        exploration_policy,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


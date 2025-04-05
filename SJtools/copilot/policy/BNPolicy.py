import torch as th
from gym import spaces
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

# Custom actor (pi) and value function (vf) networks

class BNPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    network gets sent to device automatically when proper parameter is passed to PPO.__init__ or PPO.load

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param net_arch: [16,{pi:[32,32],vf:[32,32]}] defines what net arch should look like
    
    """


    def __init__(
        self,
        feature_dim: int,
        net_arch = dict(pi=[64, 64], vf=[64, 64]),
        activation_fn = nn.Tanh,
        **kwargs
    ):
        super().__init__()
        
        # separate shared net and pv_net
        if isinstance(net_arch, list) and len(net_arch) > 0:
            # structure is [shared,dict(p&v)]
            shared_layers_dims = net_arch[:-1]
            pv_net_arch = net_arch[-1]
        else:
            # structure is dict(p&v)
            # this means no shared layer
            shared_layers_dims = []
            pv_net_arch = net_arch


        # get dimensions of policy and value nets
        if isinstance(pv_net_arch, dict):
            layers_dims_pi = pv_net_arch.get("pi", [])  # Layer sizes of the policy network
            layers_dims_vf = pv_net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            layers_dims_pi = layers_dims_vf = net_arch

        # prepare shared net
        last_layer_dim___shared = feature_dim
        shared_net = []
        for i, curr_layer_dim in enumerate(shared_layers_dims):
            shared_net.append(nn.Linear(last_layer_dim___shared, curr_layer_dim))
            shared_net.append(activation_fn())
            if i + 1 < len(shared_layers_dims): shared_net.append(nn.BatchNorm1d(curr_layer_dim))
            last_layer_dim___shared = curr_layer_dim
        self.shared_net = nn.Sequential(*shared_net)
        feature_dim = last_layer_dim___shared
        last_layer_dim_pi = last_layer_dim_vf = feature_dim
        
        
        policy_net = []
        # Iterate through the policy layers and build the policy net
        for i, curr_layer_dim in enumerate(layers_dims_pi):
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            if i + 1 < len(layers_dims_pi): policy_net.append(nn.BatchNorm1d(curr_layer_dim))
            last_layer_dim_pi = curr_layer_dim
            
        value_net = []
        # Iterate through the value layers and build the value net
        for i, curr_layer_dim in enumerate(layers_dims_vf):
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            if i + 1 < len(layers_dims_pi): value_net.append(nn.BatchNorm1d(curr_layer_dim))
            last_layer_dim_vf = curr_layer_dim

        # Policy network
        self.policy_net = nn.Sequential(*policy_net)
        # Value network
        self.value_net = nn.Sequential(*value_net)

        # IMPORTANT:
        # Save output dimensions
        # house keeping needed by stable baselines
        self.latent_dim_pi = layers_dims_pi[-1] # used by policies.py
        self.latent_dim_vf = layers_dims_vf[-1] # used by policies.py


    def forward(self, features: th.Tensor):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)
        # return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))
        # return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))
        # return self.value_net(features)
    

class BNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        print(f"Using BNPolicy with obs {observation_space.shape[0]}, and action {action_space.shape[0]}")
        print(f"net_arch: {self.net_arch}")

    def _build_mlp_extractor(self) -> None:
        # these parameters are derived from parent: self.features_dim, net_arch=self.net_arch
        self.mlp_extractor = BNPolicyNetwork(self.features_dim, net_arch=self.net_arch)

import torch as th
from gym import spaces
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

# Custom actor (pi) and value function (vf) networks


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, input_dim: int, features_dim: int = 256):
        super().__init__(input_dim, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.i_dim = n_input_channels = int(input_dim**0.5)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                torch.zeros((2,1,self.i_dim,self.i_dim)).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, X: th.Tensor) -> th.Tensor:
        # print("obs",observations.shape)
        # observations = self.reshape(observations)
        # print("obs",observations.shape)
        return self.linear(self.cnn(X))
    
    # def reshape(self,observations):
    #     return self.observations.view(self.i_dim, self.i_dim)

    
class CNNPolicyNetwork(nn.Module):
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
        
        self.heatmap_n = kwargs['heatmap_n']
        self.conv_input_dim = self.heatmap_n ** 2
        self.conv_output_dim = 36

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
        last_layer_dim___shared = feature_dim - self.conv_input_dim
        shared_net = []
        for i, curr_layer_dim in enumerate(shared_layers_dims):
            shared_net.append(nn.Linear(last_layer_dim___shared, curr_layer_dim))
            shared_net.append(activation_fn())
            if i + 1 < len(shared_layers_dims): shared_net.append(nn.BatchNorm1d(curr_layer_dim))
            last_layer_dim___shared = curr_layer_dim

        print('shared_net',shared_net)
        self.shared_net = nn.Sequential(*shared_net)
        self.conv_net = CustomCNN(input_dim=self.conv_input_dim, features_dim = self.conv_output_dim)

        feature_dim = last_layer_dim___shared + self.conv_output_dim
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

        print('features',features.shape)
        n2 = self.heatmap_n * self.heatmap_n
        heatmap = features[:,-n2:].view(-1,1,self.heatmap_n,self.heatmap_n)
        featuresWOHeatmap = features[:,:-n2]
        
        shared_latent = self.shared_net(featuresWOHeatmap)
        print('shared_latent',shared_latent.shape)

        conv_latent = self.conv_net(heatmap)
        print('conv_latent',conv_latent.shape)

        comb_shared_latent = torch.cat((shared_latent,conv_latent),1)
        
        # out = torch.cat((out1,y,z),1)
        return self.policy_net(comb_shared_latent), self.value_net(comb_shared_latent)
        # return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))
        # return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))
        # return self.value_net(features)
    

class CNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.heatmap_n = kwargs['heatmap_n']
        del kwargs['heatmap_n']

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        print(f"Using CNNPolicy with obs {observation_space.shape[0]}, and action {action_space.shape[0]}")
        print(f"net_arch: {self.net_arch}")

    def _build_mlp_extractor(self) -> None:
        # these parameters are derived from parent: self.features_dim, net_arch=self.net_arch
        self.mlp_extractor = CNNPolicyNetwork(self.features_dim, net_arch=self.net_arch, heatmap_n=self.heatmap_n)

if __name__ == '__main__':
    import numpy as np
    import torch
    from stable_baselines3.common.utils import constant_fn
    observation_space = spaces.Box(low=-1, high=1, shape= (140,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    # torch.optim.lr_scheduler.ReduceLROnPlateau
    
    model = CNNPolicy(observation_space, action_space, lr_schedule=constant_fn(0.001),heatmap_n=11) #,feature_dim=133,net_arch=[[64,64],[64,64]])
    from torchsummary import summary

    summary(model, input_size=(140,))
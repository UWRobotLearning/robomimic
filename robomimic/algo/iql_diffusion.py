"""
Implementation of Implicit Q-Learning (IQL).
Based off of https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py.
(Paper - https://arxiv.org/abs/2110.06169).
"""
import numpy as np
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.models.obs_nets as ObsNets

from typing import Callable, Union
from packaging.version import parse as parse_version
from enum import Enum
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

from robomimic.algo import register_algo_factory_func, ValueAlgo, PolicyAlgo

from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class MultiStepMethod(Enum):
    REPEAT = 'repeat'
    ONE_STEP = 'one_step'
    MULTI_STEP = 'multi_step'

@register_algo_factory_func("iql_diffusion")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the IQL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return IQLDiffusion, {}


class IQLDiffusion(PolicyAlgo, ValueAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Networks for this algo: critic (potentially ensemble), actor, value function
        """
        try:
            self.multi_step_method = MultiStepMethod(self.algo_config.multi_step_method)
        except RuntimeError:
            self.multi_step_method = MultiStepMethod.ONE_STEP

        # Create nets
        self.nets = nn.ModuleDict()
        
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        self.nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })

            
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=self.ac_dim,
        #     obs_dim=0 if (obs_as_cond) else obs_feature_dim,
        #     max_n_obs_steps=self.algo_config.horizon.observation_horizon,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )

        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()

        # Critics
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            for net_list in (self.nets["critic"], self.nets["critic_target"]):
                critic = ValueNets.ActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.ac_dim,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=self.goal_shapes,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                )
                net_list.append(critic)

        # Value function network
        self.nets["vf"] = ValueNets.ValueNetwork(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        
        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(parameters=self.nets.parameters(), power=self.algo_config.ema.power)


        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic, critic_target in zip(self.nets["critic"], self.nets["critic_target"]):
                TorchUtils.hard_update(
                    source=critic,
                    target=critic_target,
                )

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out relevant info and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        input_batch = dict()
        
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch["obs"] = {k: batch["obs"][k][:, :(To + Tp - 1), :] for k in batch["obs"]}
        # TODO: check this
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, :To, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, (To - 1):(Tp + To - 1), :]
        input_batch["dones"] = batch["dones"][:, (To - 1):(Tp + To - 1)]
        input_batch["rewards"] = batch["rewards"][:, (To - 1):(Tp + To - 1)]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        Tp = self.algo_config.horizon.prediction_horizon
        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            info = super().train_on_batch(batch, epoch, validate=validate)

            if self.multi_step_method == MultiStepMethod.REPEAT:
                q_preds = []
                v_preds = []
                for i in range(Tp):
                    # Compute loss for critic(s)
                    critic_losses, vf_loss, critic_info = self._compute_critic_loss(batch, i)
                    q_preds.append(critic_info["vf/q_pred"])
                    v_preds.append(critic_info["vf/v_pred"])
                    
                    if not validate:
                        # Critic update
                        self._update_critic(critic_losses, vf_loss)
                
                # Compute loss for actor
                actor_loss, actor_info = self._compute_actor_loss(batch, q_preds, v_preds)

                if not validate:                
                    # Actor update
                    self._update_actor(actor_loss)
            elif self.multi_step_method == MultiStepMethod.ONE_STEP:
                # Compute loss for critic(s)
                critic_losses, vf_loss, critic_info = self._compute_critic_loss(batch)
                q_pred = critic_info["vf/q_pred"]
                v_pred = critic_info["vf/v_pred"]
                # Compute loss for actor
                actor_loss, actor_info = self._compute_actor_loss(batch, q_pred, v_pred)

                if not validate:
                    # Critic update
                    self._update_critic(critic_losses, vf_loss)
                    
                    # Actor update
                    self._update_actor(actor_loss)
            else:
                raise NotImplementedError

            # Update info
            info.update(actor_info)
            info.update(critic_info)

        # Return stats
        return info

    def _compute_critic_loss(self, batch, i=0):
        """
        Helper function for computing Q and V losses. Called by @train_on_batch

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            i (int): timestep

        Returns:
            critic_losses (list): list of critic (Q function) losses
            vf_loss (torch.Tensor): value function loss
            info (dict): dictionary of Q / V predictions and losses
        """
        info = OrderedDict()
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon
            
        # get batch values
        obs = {k: batch["obs"][k][:, To - 1 + i, :] for k in batch["obs"]}
        actions = batch["actions"][:, i, :]
        next_obs = {k: batch["next_obs"][k][:, To - 1 + i, :] for k in batch["next_obs"]}
        goal_obs = batch["goal_obs"]
        rewards = torch.unsqueeze(batch["rewards"][:, i], 1)
        dones = torch.unsqueeze(batch["dones"][:, i], 1)            

        # Q predictions
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                   for critic in self.nets["critic"]]

        info["critic/critic1_pred"] = pred_qs[0].mean()

        # Q target values
        target_vf_pred = self.nets["vf"](obs_dict=next_obs, goal_dict=goal_obs).detach()
        q_target = rewards + (1. - dones) * self.algo_config.discount * target_vf_pred
        q_target = q_target.detach()

        # Q losses
        critic_losses = []
        td_loss_fcn = nn.SmoothL1Loss() if self.algo_config.critic.use_huber else nn.MSELoss()
        for (i, q_pred) in enumerate(pred_qs):
            # Calculate td error loss
            td_loss = td_loss_fcn(q_pred, q_target)
            info[f"critic/critic{i+1}_loss"] = td_loss
            critic_losses.append(td_loss)

        # V predictions
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                        for critic in self.nets["critic_target"]]
        q_pred, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)
        q_pred = q_pred.detach()
        vf_pred = self.nets["vf"](obs_dict=obs, goal_dict=goal_obs)
        
        # V losses: expectile regression. see section 4.1 in https://arxiv.org/pdf/2110.06169.pdf
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.algo_config.vf_quantile + vf_sign * (1 - self.algo_config.vf_quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        
        # update logs for V loss
        info["vf/q_pred"] = q_pred
        info["vf/v_pred"] = vf_pred
        info["vf/v_loss"] = vf_loss

        # Return stats
        return critic_losses, vf_loss, info

    def _update_critic(self, critic_losses, vf_loss):
        """
        Helper function for updating critic and vf networks. Called by @train_on_batch

        Args:
            critic_losses (list): list of critic (Q function) losses
            vf_loss (torch.Tensor): value function loss
        """

        # update ensemble of critics
        for (critic_loss, critic, critic_target, optimizer) in zip(
                critic_losses, self.nets["critic"], self.nets["critic_target"], self.optimizers["critic"]
        ):
            TorchUtils.backprop_for_loss(
                net=critic,
                optim=optimizer,
                loss=critic_loss,
                max_grad_norm=self.algo_config.critic.max_gradient_norm,
                retain_graph=False,
            )

            # update target network
            with torch.no_grad():
                TorchUtils.soft_update(source=critic, target=critic_target, tau=self.algo_config.target_tau)

        # update V function network
        TorchUtils.backprop_for_loss(
            net=self.nets["vf"],
            optim=self.optimizers["vf"],
            loss=vf_loss,
            max_grad_norm=self.algo_config.critic.max_gradient_norm,
            retain_graph=False,
        )

    def _compute_actor_loss(self, batch, q_pred, v_pred):
        """
        Helper function for computing actor loss. Called by @train_on_batch

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            critic_info (dict): dictionary containing Q and V function predictions,
                to be used for computing advantage estimates

        Returns:
            actor_loss (torch.Tensor): actor loss
            info (dict): dictionary of actor losses, log_probs, advantages, and weights
        """
        B = batch['actions'].shape[0]        
    
        info = OrderedDict()
        actions = batch['actions']

        To = self.algo_config.horizon.observation_horizon
        
        # encode obs
        inputs = {
            'obs': {k: batch["obs"][k][:, :To, :] for k in batch["obs"]},
            'goal': batch["goal_obs"]
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        
        obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]

        obs_cond = obs_features.flatten(start_dim=1)
        
        # inpainting mask
        # condition_mask = self.mask_generator(actions.shape)
        
        # sample noise to add to actions
        noise = torch.randn(actions.shape, device=self.device)
        
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=self.device
        ).long()
        
        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, timesteps)
        
        # # compute loss mask
        # loss_mask = ~condition_mask
        
        # # apply conditioning
        # noisy_actions[condition_mask] = actions[condition_mask]
        # predict the noise residual
        noise_pred = self.nets['policy']['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond)
        
        # L2 loss
        bc_loss = F.mse_loss(noise_pred, noise)
        
        self.ema.step(self.nets['policy'].parameters())

        info["actor/bc_loss"] = bc_loss.mean()

        # compute advantage estimate
        if self.multi_step_method == MultiStepMethod.REPEAT:
            q_pred = torch.cat(q_pred)
            v_pred = torch.cat(v_pred)
            discounts = self.algo_config.discount ** torch.arange(len(q_pred), dtype=torch.float32, device=self.device)
            adv = torch.sum(discounts * (q_pred - v_pred))
        elif self.multi_step_method == MultiStepMethod.ONE_STEP:
            adv = q_pred - v_pred
        
        # compute weights
        weights = self._get_adv_weights(adv)

        # compute advantage weighted actor loss. disable gradients through weights
        actor_loss = (bc_loss * weights.detach()).mean()

        info["actor/loss"] = actor_loss

        # log adv-related values
        info["adv/adv"] = adv
        info["adv/adv_weight"] = weights
        
        # Return stats
        return actor_loss, info

    def _update_actor(self, actor_loss):
        """
        Helper function for updating actor network. Called by @train_on_batch

        Args:
            actor_loss (torch.Tensor): actor loss
        """

        TorchUtils.backprop_for_loss(
            net=self.nets['policy'],
            optim=self.optimizers['policy'],
            loss=actor_loss,
        )
    
    def _get_adv_weights(self, adv):
        """
        Helper function for computing advantage weights. Called by @_compute_actor_loss

        Args:
            adv (torch.Tensor): raw advantage estimates

        Returns:
            weights (torch.Tensor): weights computed based on advantage estimates,
                in shape (B,) where B is batch size
        """
        
        # clip raw advantage values
        if self.algo_config.adv.clip_adv_value is not None:
            adv = adv.clamp(max=self.algo_config.adv.clip_adv_value)

        # compute weights based on advantage values
        beta = self.algo_config.adv.beta # temprature factor        
        weights = torch.exp(adv / beta)
        # clip final weights
        if self.algo_config.adv.use_final_clip is True:
            weights = weights.clamp(-100.0, 100.0)

        # reshape from (B, 1) to (B,)
        return weights[:, 0]

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()

        log["actor/bc_loss"] = info["actor/bc_loss"].item()
        log["actor/loss"] = info["actor/loss"].item()

        log["critic/critic1_pred"] = info["critic/critic1_pred"].item()
        log["critic/critic1_loss"] = info["critic/critic1_loss"].item()

        log["vf/v_loss"] = info["vf/v_loss"].item()

        self._log_data_attributes(log, info, "vf/q_pred")
        self._log_data_attributes(log, info, "vf/v_pred")
        self._log_data_attributes(log, info, "adv/adv")
        self._log_data_attributes(log, info, "adv/adv_weight")

        return log

    def _log_data_attributes(self, log, info, key):
        """
        Helper function for logging statistics. Moodifies log in-place

        Args:
            log (dict): existing log dictionary
            log (dict): existing dictionary of tensors containing raw stats
            key (str): key to log
        """
        log[key + "/max"] = info[key].max().item()
        log[key + "/min"] = info[key].min().item()
        log[key + "/mean"] = info[key].mean().item()
        log[key + "/std"] = info[key].std().item()

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for lr_sc in self.lr_schedulers["critic"]:
            if lr_sc is not None:
                lr_sc.step()

        if self.lr_schedulers["vf"] is not None:
            self.lr_schedulers["vf"].step()

        if self.lr_schedulers["policy"] is not None:
            self.lr_schedulers["policy"].step()

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        # select network
        nets = self.nets
        # if self.ema is not None:
        #     nets = self.ema.averaged_model

        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }

        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action
    
    
# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        out
        put: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
                    
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

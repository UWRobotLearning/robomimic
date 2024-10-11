"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("discriminator")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the Discriminator algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    algo_class, algo_kwargs = Discriminator, {}

    return algo_class, algo_kwargs


class Discriminator(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets['policy'] = PolicyNets.Discriminator_MLP(
            obs_shapes=self.obs_shapes, 
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.discriminator_layer_dims
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


    def train_on_batch(self, expert_batch, subopt_batch, epoch, validate):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # print(f'expert_batch : {expert_batch.keys()}')
            info = OrderedDict()
            prediction_on_expert = self._forward_training(expert_batch)
            prediction_on_subopt = self._forward_training(subopt_batch)
            losses = self._compute_losses(preds_on_expert=prediction_on_expert, preds_on_subopt=prediction_on_subopt)

            info['preds_on_expert'] = TensorUtils.detach(prediction_on_expert)
            info['preds_on_subopt'] = TensorUtils.detach(prediction_on_subopt)


            tp = (info['preds_on_expert']['preds'] > 0.5).float().mean()
            fp = (info['preds_on_subopt']['preds'] > 0.5).float().mean()
            tn = (info['preds_on_subopt']['preds'] < 0.5).float().mean()
            fn = (info['preds_on_expert']['preds'] < 0.5).float().mean()
            
            info['tp'] = tp.item()
            info['fp'] = fp.item()
            info['tn'] = tn.item()
            info['fn'] = fn.item()

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        
        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        preds = self.nets['policy'](obs=batch['obs'], goal_dict=batch['goal_obs'])
        predictions['preds'] = preds
        return predictions
    
    def _compute_losses(self, preds_on_expert, preds_on_subopt):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        expert_target = torch.ones_like(preds_on_expert['preds']).to(self.device)
        subopt_target = torch.zeros_like(preds_on_subopt['preds']).to(self.device)

        expert_loss = nn.BCELoss()(preds_on_expert['preds'], expert_target)
        subopt_loss = nn.BCELoss()(preds_on_subopt['preds'], subopt_target)

        losses = OrderedDict()
        losses["prediction_loss"] = expert_loss + subopt_loss
        losses["expert_prediction_loss"] = expert_loss
        losses["subopt_prediction_loss"] = subopt_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["prediction_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super().log_info(info)
        log["Loss"] = info["losses"]["prediction_loss"].item()
        log["TP"] = info["tp"]
        log["FP"] = info["fp"]
        log["TN"] = info["tn"]
        log["FN"] = info["fn"]
        return log

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
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)
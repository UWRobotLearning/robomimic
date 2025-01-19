"""
Config for IQL Diffusion algorithm.
"""

from robomimic.config.base_config import BaseConfig


class IDQLConfig(BaseConfig):
    ALGO_NAME = "idql"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(IDQLConfig, self).algo_config()

        # optimization parameters        
        self.algo.optim_params.critic.learning_rate.initial = 1e-4          # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00              # L2 regularization strength

        self.algo.optim_params.vf.learning_rate.initial = 1e-4              # vf learning rate
        self.algo.optim_params.vf.learning_rate.decay_factor = 0.0          # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.vf.learning_rate.epoch_schedule = []         # epochs where LR decay occurs
        self.algo.optim_params.vf.regularization.L2 = 0.00                  # L2 regularization strength
        
        self.algo.optim_params.policy.learning_rate.initial = 1e-4          # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.0      # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []     # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00              # L2 regularization strength
        self.algo.optim_params.policy.freeze_until_epoch = None             # don't train policy until this epoch

        self.algo.multi_step_method = 'one_step'
        self.algo.use_bc = False

        # unet params
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        
        self.algo.horizon.observation_horizon = 1
        self.algo.horizon.action_horizon = 1
        self.algo.horizon.prediction_horizon = 1
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'

        # target network related parameters
        self.algo.discount = 0.99                                           # discount factor to use
        self.algo.target_tau = 0.01                                         # update rate for target networks

        # ================== Critic Network Config ===================
        # critic ensemble parameters
        self.algo.critic.ensemble.n = 2                                     # number of Q networks in the ensemble
        self.algo.critic.layer_dims = (300, 400)                            # critic MLP layer dimensions
        self.algo.critic.use_huber = False                                  # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = None                           # L2 gradient clipping for actor

        # ================== Adv Config ==============================
        self.algo.adv.clip_adv_value = None                                 # whether to clip raw advantage estimates
        self.algo.adv.beta = 1.0                                            # temperature for operator
        self.algo.adv.use_final_clip = True                                 # whether to clip final weight calculations

        self.algo.vf_quantile = 0.9                                         # quantile factor in quantile regression

        self.algo.bottleneck_policy = False
        self.algo.bottleneck_value = False
        self.algo.q_bottleneck_beta = 0.2
        self.algo.policy_bottleneck_beta = 0.2
        self.algo.spectral_norm_policy = False
        self.algo.spectral_norm_value = False

        self.algo.lipschitz = False
        self.algo.lipschitz_constant = 3.0
        self.algo.lipschitz_weight = 0.005
        self.algo.lipschitz_slack = False
        self.algo.lipschitz_denoiser = False
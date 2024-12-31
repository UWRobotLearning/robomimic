from robomimic.config.config import Config
from robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from robomimic.config.bc_config import BCConfig
from robomimic.config.bcq_config import BCQConfig
from robomimic.config.cql_config import CQLConfig
from robomimic.config.iql_config import IQLConfig
from robomimic.config.gl_config import GLConfig
from robomimic.config.hbc_config import HBCConfig
from robomimic.config.iris_config import IRISConfig
from robomimic.config.td3_bc_config import TD3_BCConfig
from robomimic.config.idql_config import IDQLConfig
from robomimic.config.idql_qsm_config import IDQL_QSMConfig
from robomimic.config.iql_diffusion_config import IQLDiffusionConfig
from robomimic.config.discriminator_config import DiscriminatorConfig
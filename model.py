# Convert Text to Image

# Py
from enum import Enum

# PyTorch
import torch
import torch.nn
import numpy as np
from omegaconf import OmegaConf
# PyTorch Optimizations
from pytorch_lightning import seed_everything

# Diffusion Samplers
import k_diffusion as kdiff
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Utils
from ldm.util import instantiate_from_config






# Config
class SamplerTypes(Enum):
    # Comvis LDM Diffusion
    PLMS = "plms"
    DDIM = "ddim"
    # K-Diffusion
    EULER = "k_euler"
    EULER_A = "k_euler_a"
    LMS = "k_lms"
    HEUN = "k_heun"
    HEUN_A = "k_heun_a"
    DPM_2 = "k_dpm_2"
    DMP_2_A = "k_dpm_2_a"




class Generator(nn.Module):

    def __init__(self, verbose=False):
        """
            Init a model from mode_path (.ckpt) and vae_path (.vae.pt)
        """
        # Parameters
        self.verbose = verbose
        # Device (use cuda or else use cpu)
        if config.device:
            self.device(config.device)
        else:
            self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _load_config(self, config_path:str):
        """Load a config file

        Args:
            config_path (str): Path to model config file
        """
        self.config = OmegaConf.load(config_path)
        return self.config
    
    def _load_vae(self, model:any, vae_path:str):
        """Loads vae weights to the model

        Args:
            model (any): PyTorch target Model
            vae_path (str): path to .vae.pt file
        """
        # Load vae checkpoint into "cpu"
        vae_ckpt = torch.load(vae_path, map_location="cpu")
        # Find and delete loss keys
        loss = []
        for i in vae_ckpt["state_dict"].keys():
            if i[0:4] == "loss":
                loss.append(i)
        for i in loss:
            # remove loss keys from vae_checkpoint
            del vae_ckpt["state_dict"][i]
        # Load the model
        model.first_stage_model = model.first_stage_model.float()
        model.first_stage_model.load_state_dict(vae_ckpt["state_dict"])
        model.first_stage_model = model.first_stage_model.float()
        # delete
        del vae_ckpt
        del loss

    def load_model(self, config_path:str, model_path:str, vae_path:str, cpu_assisted_load=False):
        """Load a Model

        Args:
            config (str): Model Config.yml
            model_path (str): Model file (.ckpt)
            vae_path (str): Vae file (.vae.ckpt)
        """
        # Check parameters
        model_map_location = "cpu" if cpu_assisted_load else self.device
        # Load Model
        if self.verbose: print(f"Loading Model from {model_path}")
        t_ckpt = torch.load(config_path, model_path, map_location=model_map_location)
        if "global_step" in t_ckpt:
            if self.verbose: print(f"Global Step: {t_ckpt['global_step']}")
        state = t_ckpt["state_dict"]
        # Init the Model using Config
        model = instantiate_from_config(self._load_config(config_path))
        m, u = model.load_state_dict(state, strict=False)
        if (len(m) > 0 and self.verbose):
            print(f"[load_model] missing keys: {f}")
        if (len(u) > 0 and self.verbose):
            print(f"[load_model] unexpected keys: {f}")
        # Switch to cuda
        if torch.cuda.is_available(): model.cuda()
        # Evaluate Model and return
        model.eval()
        return model

    
    def load_sampler(self):
        """
            Load the sampler
        """
        if isinstance(self.config.sampler, SamplerTypes.PLMS):
            self.sampler = PLMSSampler(self.)

    def _sample(self, *args, **kwargs):
        """
            Sampler sample interface method. To be set from code

        """
        pass

    @torch.no_grad
    def generate(batch_size: int, steps: int, cfg: float, height: int, width: int, prompt: str or torch.Tensor, negative_prompt: str or torch.Tensor):
        # Generate

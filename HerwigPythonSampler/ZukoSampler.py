import torch
import zuko
from zuko.flows import Flow, UnconditionalDistribution, UnconditionalTransform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.distributions import DiagNormal, BoxUniform
from zuko.transforms import RotationTransform

class ZukoSampler:
    def __init__(self, dims=0, uniform_latent=True):

        flow = Flow(
            transform=[
                MaskedAutoregressiveTransform(dims, 0, hidden_features=(64, 64)),
                UnconditionalTransform(RotationTransform, torch.randn(dims, dims)),
                MaskedAutoregressiveTransform(dims, 0, hidden_features=(64, 64)),
            ],
            base=UnconditionalDistribution(
                BoxUniform,
                torch.zeros(dims),
                torch.ones(dims),
                buffer=True,
            ),
        )
        self.model = flow
        # self.model = zuko.flows.NSF(features=dims, transforms=3, hidden_features=(64, 64))

    
    def sample(
        self,
        n: int | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        c: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
    
        if return_prob or return_log_prob:
            # Use the built-in method that returns both samples and log probabilities
            x, log_prob = self.model().rsample_and_log_prob(torch.Size([n]))
            
            if return_prob:
                prob = torch.exp(log_prob)
                return x, prob
            else:
                return x, log_prob
        else:
            # Just sample without probabilities
            x = self.model.rsample(torch.Size([n]))
            return x
    
    def log_prob(
        self,
        x):
        return self.model().log_prob(x), 0
    
    def eval(self):
        self.model.eval()
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self):
        self.model.train()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.model.state_dict()
    

        
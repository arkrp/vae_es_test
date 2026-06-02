#section-start setup
print("Hello World!")
import math
import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F, init, Module
from torch.nn.parameter import Parameter, Uninitializ
from Datasets import MNIST
print("preload complete")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded:" + repr(device))
#section-end
class EggLinear(Module): #section-start
    #section-start """
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = EggLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    #section-end
    #TODO make the forward use the perterbations.
    #TODO implement the egg_grad function.
    #TODO STRETCH make the perturb function perturb symetrically.
    #TODO ensure all of your tensors are full dimensional to avoid broadcast ambiguity.
    #section-start attributes
    __constants__ = ["in_features", "out_features"]
    in_features: int # m
    out_features: int # n
    weight: Tensor # m x n
    perturbation_rank: int # r
    perturbation_stdev: float # sigma
    A_perturbation: Tensor # batch_size x m x r
    B_perturbation: Tensor # batch_size x n x r
    bias_perturbation: Tensor # batch_size x n
    #section-end
    def __init__( #section-start
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        perturbation_rank: int
        perturbation_stdev: float
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.perturbation_rank = perturbation_rank
        self.perturbation_stdev = perturbation_stdev
    #section-end
    def reset_parameters(self) -> None: #section-start
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    #section-end
    def forward(self, input: Tensor) -> Tensor: #section-start
        """
        Runs the forward pass.
        """
        return F.linear(input, self.weight, self.bias)
    #section-end
    def extra_repr(self) -> str: #section-start
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    #section-end
    def perturb(self, batch_size) -> None: #section-start
        #section-start """
        """
        Sample the perturbation matricies/vectors for a eggroll learning step.
        """
        #section-end
        #section-start set A perturbation
        self.A_perturbation = torch.normal(torch.zeros(
            size=(
                batch_size,
                self.in_features,
                self.perturbation_rank
                )
            ), std=1)
        #section-end
        #section-start set B perturbation
        self.B_perturbation = torch.normal(torch.zeros(
            size=(
                batch_size,
                self.out_features,
                self.perturbation_rank
                )
            ), std=1)
        #section-end
        #section-start set bias perturbation if applicable
        if self.bias is not None:
            self.bias_perturbation=torch.normal(torch.zeros(
                size=(
                    batch_size,
                    self.out_features
                    )
            ), std=1)
        #section-end
    #section-end
    def egg_grad(self, loss_scores): #section-start
        #section-start """
        """
        Uses the batchwise scores to produce the gradient estimates!
        arguments:
            loss_scores: Tensor: (batch_size)
        """
        #section-end
        weight.grad = (
            (1/self.perturbation_stdev)*
            torch.repeat_interleave(loss_scores)
        )
        #TODO
            
    #section-end
#section-end
#section-start ending phrase
print("Serpent praise")
#section-end

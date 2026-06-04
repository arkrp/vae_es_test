#section-start notes

    #TODO move todo list to NOTES file NOTES

    #TODO finish EggLinear
        #DONE make the forward use the perterbations.
        #DONE implement the egg_grad function.
        #STRETCH make the perturb function perturb symetrically.

    #TODO create EggBatchNorm

    #TODO Make Egg optimizer
    #TODO Validate egg optimizer

    #TODO Ensure Egg classes do not interfere with regular gradient decent.

    #TODO run VAE Test

    #RULE all tensors used in broadcast operations must have an equal dimensionality.
    #RULE assert check dimensions after complicated matrix operaions.

    #NOTE torch computes its forward with y = xM^{T} + b. This is admittedly good design, but I would like to clarify that my perterbations are relative to M rather than to M^{T} for this reason I am going do some transposing to make the adding perterbation in the forward pass work. Specifically
    #y = ((M + sE) @ x^{T})^{T} + b
    #y = (M @ x^{T})^{T} + s(E @ x^{T})^{T} + b
    #y = x@(M^{T}) + s(E @ x^{T})^{T} + b
    #y = F.linear(x, M, s(E @ x^{T})^{T} + b))
    #Where E is the perturbation, M is the weight matrix, and s is the perturbation standard deviation.

    #TODO PARTIALLY DONE Recheck math. The negative sign in the math doesn't make intuitive sense. Isn't that the wrong direction?
    #UPDATE yeah the paper doesn't have the negative in it. I mist have messed up somewhere. Ah well. Don't really matter. I'm just gonna remove the negative

#section-end
#section-start setup
print("Hello World!")
import math
import torch
import torchvision
from torch import Tensor, Shape
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
    #section-start attributes
    __constants__ = ["in_features", "out_features"]
    in_features: int # m
    out_features: int # n
    weight: Tensor # m x n
    E_perturbation_rank: int # r
    E_perturbation_stdev: float # sigma
    E_A_perturbation: Tensor # batch_size x m x r
    E_B_perturbation: Tensor # batch_size x n x r
    bias_perturbation: Tensor # batch_size x n
    bias_perturbation_stdev: float # sigma
    #section-end
    def __init__( #section-start
        #section-start args
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        E_perturbation_rank: int,
        E_perturbation_stdev: float,
        bias_perturbation_stdev: float
    ) -> None:
        #section-end
        #section-start initialize the primary attributes
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
        #section-end
        #section-start load perturbation based attributes
        self.E_perturbation_rank = E_perturbation_rank
        self.E_perturbation_stdev = E_perturbation_stdev
        self.bias_perturbation_stdev = bias_perturbation_stdev
        #section-end
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
        return F.linear(
            input,
            self.weight,
            multiply_by_egg(
                A_perturbation=self.A_perturbation,
                B_perturbation=self.B_perturbation,
                vector=input) +
            self.bias)
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
    def egg_grad(self, loss): #section-start
        #section-start """
        """
        shoves the egg gradient estimates into the parameter gradients so the torch optimizers can fiddle with em.
        arguments:
            loss: Tensor: (batch_size)
        """
        self.weight.grad = bake_matrix_perturbation(
            A_perturbation=self.A_perturbation,
            B_perturbation=self.B_perturbation,
            perturbation_stdev=self.perturbation_stdev,
            loss=loss)
        if(self.bias is not None):
            self.bias.grad = bake_vector_perturbation(
                perturbation=self.bias_perturbation,
                perturbation_stdev=self.bias_perturbation_stdev,
                loss=loss)
        #section-end
#section-end
#section-end
def bake_matrix_perturbation( #section-start
    #section-start args
    *,
    A_perturbation,
    B_perturbation,
    perturbation_stdev,
    loss
):
    #section-end
    #section-start """
    """
    turns a batch of matrix perturbations and their associated losses into an estimate of the loss gradient around the perturbed point.

    This corresponds to the gradient estimator step in the associated documentation.
    
    Args:

    A_perturbation - Tensor batch_size x m x r
    B_perturbation - Tensor batch_size x n x r
    perturbation_stdev - float
    loss - Tensor batch_size
    """
    #section-end
    #section-start arange size validation
    #section-start record input sizes
    batch_size = A_perturbation.shape[0]
    m = A_perturbation.shape[1]
    r = A_perturbation.shape[2]
    n = B_perturbation.shape[1]
    #section-end
    #section-start validate input sizes
    assert len(A_perturbation.size) == 3
    assert B_perturbation.shape[0] == batch_size
    assert B_perturbation.shape[2] == r
    assert len(B_perturbation.size) == 3
    assert loss.shape[0] == batch_size
    assert len(loss.shape) == 1
    #section-end
    #section-end
    #section-start make intermediate matricies
    #section-start produce the A megamatrix
    #section-start construct matrix
    A_megamatrix = A_perturbation.permute(1, 0, 2)
    A_megamatrix = A_megamatrix.flatten(start_dim=1, end_dim=2)
    #section-end
    #section-start validate matrix size
    assert A_megamatrix.shape[0] == m
    assert A_megamatrix.shape[1] == r * batch_size
    assert len(A_megamatrix.shape) == 2
    #section-end
    #section-end
    #section-start produce the B megamatrix
    #section-start construct matrix
    B_megamatrix = B_perturbation.permute(0, 2, 1)
    B_megamatrix = B_megamatrix.flatten(start_dim=0, end_dim=1)
    #section-end
    #section-start validate matrix size
    assert B_megamatrix.shape[0] == r * batch_size
    assert B_megamatrix.shape[1] == n
    assert len(B_megamatrix.shape) == 2
    #section-end
    #section-end
    #section-start produce the loss matrix
    #section-start construct matrix
    loss_matrix = loss.repeat_interleave(r).reshape((1,-1))
    #section-end
    #section-start validate matrix size
    assert loss_matrix.shape[0] == 1
    assert loss_matrix.shape[1] == r * batch_size
    assert len(loss_matrix.shape) == 2
    #section-end
    #section-end
    #section-end
    #section-start produce gradient estimate
    #section-start compute matrix
    gradient_estimate = (
        (1.0/(batch_size*perturbation_stdev*torch.sqrt(r))) *
        (
            (A_megamatrix*loss_matrix) @
            B_megamatrix
        )
    )
    #section-end
    #section-start validate matrix size
    assert gradient_estimate.size[0] = m
    assert gradient_estimate.size[1] = n
    assert len(gradient_estimate.size) == 2
    #section-end
    #section-end
    #section-start return gradient estimate
    return gradient_estimate
    #section-end
#section-end
def bake_vector_perturbation( #section-start
    #section-start args
    perturbation,
    perturbation_stdev,
    loss
):
    #section-end
    #section-start """
    """
    turns a batch of vector perturbations and their associated losses into an estimate of the loss gradient around the perturbed  point.

    This corresponds to the gradient estimator step in the associated documention.

    Args:

    perturbation - Tensor batch_size x n
    perturbation_stdev - float
    loss - Tensor batchsize
    """
    #section-end
    #section-start set up validation
    batch_size = perturbation.size[0]
    n = perturbation.size[1]
    assert len(perturbation.size) == 2
    assert loss_scores.size[0] == batch_size
    assert len(loss_scores.size) == 1
    #section-end
    #section-start produce the loss matrix
    #section-start construct matrix
    loss_matrix = loss.reshape((-1,1))
    #section-end
    #section-start validate matrix size
    assert loss_matrix.shape[0] == 1
    assert loss_matrix.shape[1] == batch_size
    assert len(loss_matrix.size) == 2
    #section-end
    #section-end
    #section-start make the gradient estimate
    gradient_estimate = (
        (1.0/(perturbation_stdev*batch_size)) *
        (loss_matrix * perturbation).sum(dim=0)
    )
    #section-start validate matrix size
    assert gradient_estimate.size[0] == n
    assert len(gradient_estimate.size) == 1
    #section-end
    #section-end
    #section-start return gradient estimate
    return gradient_estimate
    #section-end
#section-end
def multiply_by_egg( #section-start
#section-start args
    *,
    A_perturbation,
    B_perturbation,
    vector
):
#section-end
#section-start """
"""
calculates E@x. As in the matrix E times the vector x. Takes in x as a one dimensional vector instea of a 2 so I guess it really computse E@x^{T}? yeah. but then it gives it back as a one vector so its really (E@x^{T})^{T} which is exacly what the equation calls for so good.

Args
A_perturbation - Tensor batch_size x m x r
B_perturbation - Tensor batch_size x n x r
vector - Tensor batch_size x n
"""
#section-end
    #section-start set up validation
    #section-start intuit input size
    batch_size = A_perturbutaion.size[0]
    m = A_perturbation.size[1]
    r = A_perturbation.size[2]
    n = B_Perturbation.size[1]
    #section-end
    #section-start validate input size
    assert len(A_perturbation.size) == 3
    assert B_perturbation.size[0] == batch_size
    assert B_perturbation.size[2] == r
    assert len(B_perturbation.size) == 3
    assert vector.size[0] == batch_size
    assert vector.size[1] == n
    assert len(vector.size) == 2
    #section-end
    #section-end
    #section-start calculate result
    column_vector = vector.reshape(batch_size, n, 1)
    B_perturbation_transpose = B_permutation.permute(0,2,1)
    result = A_perturbation@(B_perturbation_transpose@column_vector)
    flattened_result = result.reshape(batch_size, m)
    #section-end
    #section-start return result
    return flattened_result
    #section-end
#section-end
#section-start ending phrase
print("Serpent praise")
#section-end

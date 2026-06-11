#section-start import stuff
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init, Module, BatchNorm1d
import math
import torch
from torch import Tensor
#section-end
DEFAULT_PERTURBATION_STDEV = 1
#section-start define modules
class EggVector(Module): #section-start
    #section-start """
    """
    just a vector which is capable of being eggroll gradient updated
    """
    #section-end
    #section-start attribute
    num_features: int
    vector: Tensor
    _perturbation: Tensor
    _perturbation_stdev: float
    #section-end
    def __init__(self, *, num_features, perturbation_stdev=DEFAULT_PERTURBATION_STDEV): #section-start
        super().__init__()
        self.num_features = num_features
        self._perturbation = None
        self._perturbation_stdev = perturbation_stdev
        self.reset_parameters()
    #section-end
    def reset_parameters(self) -> None: #section-start
        self.vector = Parameter(torch.normal(
            mean=torch.zeros(self.num_features),
            std=1
        ))
    #section-end
    def forward(self, *, batch_size): #section-start
        #section-start orient parameter vector!
        retval = self.vector.reshape((1,-1))
        #section-end
        #section-start add extant pertubations
        if self._perturbation is not None:
            #section-start validate perturbation
            assert self._perturbation.shape[0] == batch_size
            assert self._perturbation.shape[1] == self.num_features
            assert len(self._perturbation.shape) == 2
            #section-end
            #section-start add perturbation to vector
            retval = (
                retval +
                self._perturbation)
            #section-end
        #section-end
        #section-start just rescale otherwise
        else:
            #section-start expand to batch size
            retval = (
                retval +
                torch.zeros(size=(
                    batch_size,
                    self.num_features)))
            #section-end
        #section-end
        #section-start validate output
        assert retval.shape[0] == batch_size
        assert retval.shape[1] == self.num_features
        assert len(retval.shape) == 2
        #section-end
        #section-start return output!
        return(retval)
        #section-end
    #section-end
    def perturb(self, batch_size) -> None: #section-start
        #section-start perturb symmetrically for even batch size
        if batch_size%2==0:
            top = torch.normal(
                mean=torch.zeros(size=(batch_size//2, self.num_features)),
                std=1.0
            )
            self._perturbation = torch.cat([top,-top], dim=0)
        #section-end
        #section-start perturb random for odd batch size
        else:
            self._perturbation = torch.normal(
                mean=torch.zeros(size=(batch_size, self.num_features)),
                std=1.0
            )
        #section-end
    #section-end
    def reset_perturbation(self) -> None: #section-start
        self._perturbation = None
    #section-end
    def egg_grad(self, loss) -> None: #section-start
        #section-start estimate the gradient
        grad_estimate = bake_vector_perturbation(
            perturbation=self._perturbation,
            perturbation_stdev=self._perturbation_stdev,
            loss=loss)
        #section-end
        #section-start write the estimate to the parameter grad!
        self.vector.grad=grad_estimate
        #section-end
    #section-end
#section-end
class EggMatrix(Module): #section-start
    #section-start """
    """
    A matrix which is capable of being eggroll gradient updated. Intended for use in matrix multiplication only.
    """
    #section-end
    #section-start attribute
    num_input_features: int
    num_output_features: int
    matrix: Tensor
    _A_perturbation: Tensor
    _B_perturbation: Tensor
    _perturbation_rank: int
    _perturbation_stdev: float
    #section-end
    def __init__(self, *, num_input_features, num_output_features, perturbation_rank=None, perturbation_stdev=DEFAULT_PERTURBATION_STDEV): #section-start
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self._A_perturbation = None
        self._B_perturbation = None
        #section-start autoset appropriate training rank if unspecified
        if(perturbation_rank==None):
            shortest_dim = max(
                [num_input_features, num_output_features])
            perturbation_rank = math.ceil(
                math.pow(
                    shortest_dim,
                    1/3))
        #section-end
        self._perturbation_rank = perturbation_rank
        self._perturbation_stdev = perturbation_stdev
        self.reset_parameters()
    #section-end
    def reset_parameters(self) -> None: #section-start
        self.matrix = Parameter(torch.normal(
            mean=torch.zeros(
                size=(
                    self.num_output_features,
                    self.num_input_features)
            ),
            std=1
        ))
    #section-end
    def forward(self, input_vector): #section-start
        #section-start """
        """
        Multiplies a vector by the matrix.
        Args:
        input_vector: Tensor batch_size x num_input_features
        """
        #section-end
        #section-start validate input
        batch_size = input_vector.shape[0]
        assert input_vector.shape[1] == self.num_input_features
        assert len(input_vector.shape) == 2
        #section-end
        #section-start shape the input vector into a column
        input_column_vector = input_vector.reshape((
                batch_size,
                self.num_input_features,
                1
        ))
        #section-end
        #section-start compute the unperturbed part
        output_column_vector = self.matrix @ input_column_vector
        #section-end
        #section-start add in the perturbation part
        if self._A_perturbation is not None:
            #section-start validate perturbation sizes
            assert self._A_perturbation.shape[0] == batch_size
            assert self._A_perturbation.shape[1] == self.num_output_features
            assert self._A_perturbation.shape[2] == self._perturbation_rank
            assert len(self._A_perturbation.shape) == 3
            assert self._B_perturbation.shape[0] == batch_size
            assert self._B_perturbation.shape[1] == self.num_input_features
            assert self._B_perturbation.shape[2] == self._perturbation_rank
            assert len(self._B_perturbation.shape) == 3
            #section-end
            #section-start do the addition
            output_column_vector = (
                output_column_vector +
                (
                    self._perturbation_stdev /
                    math.sqrt(self._perturbation_rank)
                ) *
                (
                    self._A_perturbation @
                    (
                        self._B_perturbation.transpose(1,2) @
                        input_column_vector
                    )
                )
            )
            #section-end
        #section-end
        #section-start return the result
        return (output_column_vector.
            transpose(1,2).
            reshape((
                batch_size,
                self.num_output_features)))
        #section-end
    #section-end
    def perturb(self, batch_size) -> None: #section-start
        #section-start perturb symmetrically for even batch size
        if batch_size%2==0:
            #section-start produce A
            A_top = torch.normal(
                mean=torch.zeros(size=(
                    batch_size//2,
                    self.num_output_features,
                    self._perturbation_rank
                )),
                std=1
            )
            self._A_perturbation = torch.cat(
                tensors=[A_top, -A_top],
                dim=0)
            #section-end
            #section-start produce B
            B_top = torch.normal(
                mean=torch.zeros(size=(
                    batch_size//2,
                    self.num_input_features,
                    self._perturbation_rank
                )),
                std=1
            )
            self._B_perturbation = torch.cat(
                tensors=[B_top, -B_top],
                dim=0)
            #section-end
        #section-end
        #section-start perturb random for odd batch size
        else:
            #section-start produce A
            self._A_perturbation = torch.normal(
                mean=torch.zeros(size=(
                    batch_size,
                    self.num_output_features,
                    self._perturbation_rank
                )),
                std=1
            )
            #section-end
            #section-start produce B
            self._B_perturbation = torch.normal(
                mean=torch.zeros(size=(
                    batch_size,
                    self.num_input_features,
                    self._perturbation_rank
                )),
                std=1
            )
            #section-end
        #section-end
    #section-end
    def reset_perturbation(self) -> None: #section-start
        self._A_perturbation = None
        self._B_perturbation = None
    #section-end
    def egg_grad(self, loss) -> None: #section-start
        #section-start estimate the gradient!
        grad_estimate = bake_matrix_perturbation(
            A_perturbation=self._A_perturbation,
            B_perturbation=self._B_perturbation,
            perturbation_stdev=self._perturbation_stdev,
            loss=loss)
        #section-end
        #section-start write estimate to parameter gradient!
        self.matrix.grad=grad_estimate
        #section-end
    #section-end
#section-end
class EggAffine(Module): #section-start
    #section-start """
    """
    Basic affine layer which accepts egg optimization.
    """
    #section-end
    #section-start attributes
    bias_module: EggVector
    linear_module: EggMatrix
    num_input_features: int
    num_output_features: int
    #section-end
    def __init__(self, *, num_input_features, num_output_features): #section-start
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.bias_module = EggVector(
            num_features=self.num_output_features)
        self.linear_module = EggMatrix(
            num_input_features=self.num_input_features,
            num_output_features=self.num_output_features)
    #section-end
    def forward(self, x): #section-start
        #section-start validate input
        batch_size = x.shape[0]
        assert x.shape[1] == self.num_input_features
        assert len(x.shape) == 2
        #section-end
        #section-start calculate output
        retval = self.linear_module(input_vector=x)
        assert retval is not None
        retval = retval + self.bias_module(batch_size=batch_size)
        #section-end
        #section-start return output!
        return(retval)
        #section-end
    #section-end
#section-end
class EggScaleShift(Module): #section-start
    #section-start """
    """
    Egg module to scale and shift each dimension of vector independently, used to make batchnorm (This is more restricted than EggAffine)
    """
    #section-end
    #section-start attributes!
    num_features: int
    scale_module: EggVector
    bias_module: EggVector
    #section-end
    def __init__(self, *, num_features): #section-start
        super().__init__()
        self.num_features = num_features
        self.scale_module = EggVector(
            num_features=num_features)
        self.bias_module = EggVector(
            num_features=num_features)
    #section-end
    def forward(self, x): #section-start
        batch_size = x.shape[0]
        assert x.shape[1] == self.num_features
        assert len(x.shape) == 2
        x = x * self.scale_module(batch_size=batch_size)
        x = x + self.bias_module(batch_size=batch_size)
        return(x)
    #section-end
#section-end
class EggBatchNorm1d(Module): #section-start
    #section-start """
    """
    A batchnorm layer with scaling and bias terms that are egg trainable.
    """
    #section-end
    #section-start attributes
    num_features: int
    batchnorm_layer: BatchNorm1d
    scale_shift_layer: EggScaleShift
    #section-end
    def __init__(self, num_features): #section-start
        super().__init__()
        self.batchnorm_layer = BatchNorm1d(
            num_features,
            affine=False)
        self.scale_shift_layer = EggScaleShift(
            num_features=num_features)
    #section-end
    def forward(self, x): #section-start
        x = self.batchnorm_layer(x)
        x = self.scale_shift_layer(x)
        return(x)
    #section-end
#section-end
def perturb(batch_size): #section-start
    #section-start """
    """
    This is the way to perturb modules nested in other modules. Just use model.apply(perturb(batch_size))
    """
    #section-end
    def perturb_inner(module): #section-start
        if(
            (type(module) is EggMatrix) or
            (type(module) is EggVector)):
            module.perturb(batch_size)
    #section-end
    return perturb_inner
#section-end
def egg_grad(loss): #section-start
    #section-start """
    """
    This is the way to call egg grad in modules nested in other modules. Just use model.apply(egg_grad())
    """
    #section-end
    def egg_grad_inner(module): #section-start
        if(
            (type(module) is EggMatrix) or
            (type(module) is EggVector)):
            module.egg_grad(loss)
    #section-end
    return egg_grad_inner
#section-end
def reset_perturbation(): #section-start
    #section-start """
    """
    This is the way to call reset_perturbation in modules nested in other modules. Just use model.apply(reset_perturbation())
    """
    #section-end
    def reset_perturbation_inner(module): #section-start
        if(
            (type(module) is EggMatrix) or
            (type(module) is EggVector)):
            module.reset_perturbation()
    #section-end
    return reset_perturbation_inner
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
    assert len(A_perturbation.shape) == 3
    assert B_perturbation.shape[0] == batch_size
    assert B_perturbation.shape[2] == r
    assert len(B_perturbation.shape) == 3
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
        (1.0/(batch_size*perturbation_stdev*math.sqrt(r))) *
        (
            (A_megamatrix*loss_matrix) @
            B_megamatrix
        )
    )
    #section-end
    #section-start validate matrix size
    assert gradient_estimate.shape[0] == m
    assert gradient_estimate.shape[1] == n
    assert len(gradient_estimate.shape) == 2
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
    batch_size = perturbation.shape[0]
    n = perturbation.shape[1]
    assert len(perturbation.shape) == 2
    assert loss.shape[0] == batch_size
    assert len(loss.shape) == 1
    #section-end
    #section-start produce the loss matrix
    #section-start construct matrix
    loss_matrix = loss.reshape((-1,1))
    #section-end
    #section-start validate matrix size
    assert loss_matrix.shape[0] == batch_size
    assert loss_matrix.shape[1] == 1
    assert len(loss_matrix.shape) == 2
    #section-end
    #section-end
    #section-start make the gradient estimate
    gradient_estimate = (
        (1.0/(perturbation_stdev*batch_size)) *
        (loss_matrix * perturbation).sum(dim=0)
    )
    #section-start validate matrix size
    assert gradient_estimate.shape[0] == n
    assert len(gradient_estimate.shape) == 1
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
    batch_size = A_perturbation.shape[0]
    m = A_perturbation.shape[1]
    r = A_perturbation.shape[2]
    n = B_perturbation.shape[1]
    #section-end
    #section-start validate input size
    assert len(A_perturbation.shape) == 3
    assert B_perturbation.shape[0] == batch_size
    assert B_perturbation.shape[2] == r
    assert len(B_perturbation.shape) == 3
    assert vector.shape[0] == batch_size
    assert vector.shape[1] == n
    assert len(vector.shape) == 2
    #section-end
    #section-end
    #section-start calculate result
    column_vector = vector.reshape(batch_size, n, 1)
    B_perturbation_transpose = B_perturbation.permute(0,2,1)
    result = A_perturbation@(B_perturbation_transpose@column_vector)
    flattened_result = result.reshape(batch_size, m)
    #section-end
    #section-start return result
    return flattened_result
    #section-end
#section-end
#section-end

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
from torch import Tensor
from torch.optim import Adam
from Datasets import MNIST
from EggModule import EggLinear
print("preload complete")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded:" + repr(device))
#section-end
def main(): #section-start
    #section-start print start message
    print("Starting main program")
    #section-end
    model = EggLinear(
        in_features=10,
        out_features=10,
        bias=True,
        device=None,
        dtype=None,
        E_perturbation_rank=2,
        E_perturbation_stdev=0.5,
        bias_perturbation_stdev=0.5
    )
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 32
    print("Begining grad optimization test. Loss should go down.")
    for i in range(10000):
        input_tensor = torch.normal(torch.zeros(size=(batch_size, 10)), std=1)
        desired_output_tensor = torch.arange(10).reshape(1,10)
        output_tensor = model(input_tensor)
        loss = torch.sum(torch.pow(desired_output_tensor-output_tensor,2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000==0:
            print("Iteration: (" + str(i) + ") Loss: (" + str(loss) + ")")
    print("Test Complete")
    #section-start ending phrase
    print("Serpent praise")
    #section-end
#section-end
if __name__=="__main__": #section-start
    main()
#section-end

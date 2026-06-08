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
from EggModule import EggVector, EggMatrix
print("preload complete")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded:" + repr(device))
#section-end
def grad_test(): #section-start
    model = EggLinear(
        in_features=10,
        out_features=10,
        bias=True,
        device=None,
        dtype=None,
        E_perturbation_rank=2,
        E_perturbation_stdev=0.1,
        bias_perturbation_stdev=0.1
    )
    print("Begining grad optimization test. Loss should go down.")
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 32
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
#section-end
def egg_grad_test_vector(): #section-start
    #section-start make the model
    model = EggVector(
        num_features=3,
        perturbation_stdev=0.1
    )
    #section-end
    #section-start grad decent the model
    print("Begining grad optimization test. Loss should go down.")
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 32
    desired_output_tensor = torch.arange(3)
    for i in range(20000):
        #print(list(model.parameters()))
        model.perturb(batch_size)
        output_tensor = model(batch_size=batch_size)
        loss = torch.sum(torch.pow(desired_output_tensor-output_tensor,2), dim=(1,))
        model.egg_grad(loss)
        optimizer.step()
        if i%1000==0:
            print("Iteration: (" + str(i) + ") AvgLoss: (" + str(torch.sum(loss)/batch_size) + ")")
    model.reset_perturbation()
    print("desired output: " + str(desired_output_tensor) + "\ntrained_output: " + str(model(batch_size=1)[0,]))
    print("Test Complete")
    #section-end
#section-end
def egg_grad_test_matrix(): #section-start
    #section-start define test constants
    input_size = 4
    output_size = 3
    perturbation_rank = 2
    #section-end
    #section-start make the model
    model = EggMatrix(
        num_input_features=input_size,
        num_output_features=output_size,
        perturbation_rank=perturbation_rank,
        perturbation_stdev=0.1
    )
    #section-end
    #section-start generate true matrix
    true_matrix = torch.normal(
        mean=torch.zeros(
            size=(
                output_size,
                input_size
        )),
        std=0.5)
    #section-end
    #section-start grad decent the model
    #section-start announce we are starting
    print("Begining grad optimization test. Loss should go down.")
    #section-end
    #section-start set optimization conditions
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 32
    #section-end
    #section-start loop to train!
    for i in range(20000):
        #section-start generate input
        input_tensor = torch.normal(
            mean=torch.zeros(
                size=(
                    batch_size,
                    input_size
                )
            ),
            std=1
        )
        #section-end
        #section-start generate true output
        true_output = (
            true_matrix[None, :, :] @
            input_tensor[:, :, None])
        true_output = true_output.reshape((
            batch_size,
            output_size
        ))
        #section-end
        #section-start perturb the model
        model.perturb(batch_size)
        #section-end
        #section-start generate predicted output
        output_tensor = model(input_vector=input_tensor)
        #section-end
        #section-start aquire the loss
        loss = torch.sum(torch.pow(true_output-output_tensor,2), dim=(1,))
        #section-end
        #section-start estimate gradient with loss
        model.egg_grad(loss)
        #section-end
        #section-start decend the gradient!
        optimizer.step()
        #section-end
        if i%1000==0:
            print("Iteration: (" + str(i) + ") AvgLoss: (" + str(torch.sum(loss).item()/batch_size) + ")")
    #section-end
    #section-start unperturb the model
    model.reset_perturbation()
    #section-end
    #section-start display results in readable way!
    print("Results. Matricies should look similar.")
    print("true matrix: " + str(true_matrix))
    print("trained_output: " + str(model._matrix))
    print("Test Complete")
    #section-end
    #section-end
#section-end
def main(): #section-start
    #section-start print start message
    print("Starting main program")
    #section-end
    egg_grad_test_matrix()
    #section-start ending phrase
    print("Serpent praise")
    #section-end
#section-end
if __name__=="__main__": #section-start
    main()
#section-end

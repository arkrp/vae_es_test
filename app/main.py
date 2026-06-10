#section-start import stuff
from Datasets import MNIST
import torchvision
import torch
from EggVAE import EggVAEGaussian
#section-end
def train_epoch(*, model, loss_function, training_dataloader, optimizer, batch_size, training_epoch_granularity=7, training_epoch_dots=5): #section-start
    #section-start compute numbers for batching
    data_size = len(training_dataloader.dataset)
    num_batches = ceil(data_size/batch_size)
    batch_report_point = num_batches//training_epoch_granularity
    batch_dot_point = (batch_report_point//training_epoch_dots)
    #section-end
    #section-start set model to train mode
    model.train()
    #section-end
    #section-start loop to train!
    for batch_number, (X,Y) in enumerate(training_dataloader):
        #section-start perturb the model
        model.apply(perturb(batch_size))
        #section-end
        #section-start compute the loss
        loss = loss_function(X, Y, model(X))
        #section-end
        #section-start make sure we don't get NaN losses
        if loss.isnan().any():
            raise RuntimeError(f'Nan loss occurred during training. D: {batch_number=}')
        #section-end
        #section-start estimate the gradient with egg
        optimizer.zero_grad()
        model.apply(egg_grad(loss))
        #section-end
        #section-start step the optimizer
        optimizer.step()
        #section-end
        #section-start print out progress
        report_point = batch_number % batch_report_point
        dot_point = report_point % batch_dot_point
        if batch_dot_point - dot_point == 1:
            print('.', end='', flush=True)
        if batch_report_point - report_point == 1:
            loss_score = loss.item()
            current = batch_number * batch_size + len(X)
            print(f' Average Loss: {loss_score}, Progress: {current}/{data_size}')
        #section-end
    #section-end
    #section-start place model in eval mode
    model.apply(reset_perturbation())
    model.eval()
    #section-end
#section-end
def VAE_loss_function(X, Y, ELBO): #section-start
    return(-ELBO)
#section-end
def main(): #section-start
    #section-start set training parameters
    model = EggVAEGaussian(
        data_shape=torch.Size([1, 28, 28]),
        embedding_shape=torch.Size([10]),
    )
    loss_function = VAE_loss_function
    training_dataloader, testing_dataloader = MNIST()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 32
    #section-end
    #section-start train for an epoch
    train_epoch(
        model=model,
        loss_function=loss_function,
        training_dataloader=training_dataloader,
        optimizer=optimizer,
        batch_size=batch_size)
    #section-end
#section-end
#section-start call main
if __name__ == "__main__":
    main()
#section-end

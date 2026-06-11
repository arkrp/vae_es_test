#section-start import stuff
from Datasets import MNIST
import torchvision
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
from EggModule import perturb, egg_grad, reset_perturbation
from EggVAE import EggVAEGaussian, EggSimpleNet
#section-end
# An awful note to self. Oddly image performance appears to continue to do better even after the loss stops going down. In fact when the loss hits its minimum it looks like garbage. This is kind of awful.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
def egg_train_epoch(*, model, loss_function, training_dataloader, optimizer, batch_size, training_epoch_granularity=7, training_epoch_dots=5): #section-start
    with torch.no_grad():
        #section-start compute numbers for batching
        data_size = len(training_dataloader.dataset)
        num_batches = math.ceil(data_size/batch_size)
        batch_report_point = num_batches//training_epoch_granularity
        batch_dot_point = (batch_report_point//training_epoch_dots)
        #section-end
        #section-start set model to train mode
        model.train()
        #section-end
        #section-start loop to train!
        for batch_number, (X,Y) in enumerate(training_dataloader):
            #section-start move data to the correct device
            X = X.to(device)
            Y = Y.to(device)
            #section-end
            #section-start perturb the model
            actual_batch_size = X.shape[0]
            model.apply(perturb(actual_batch_size))
            #section-end
            #section-start compute the loss
            #model_output = model(X)#TODO restore
            model_output = model(torch.zeros(size=[actual_batch_size,1]))#TODO delete
            loss = loss_function(X, Y, model_output)
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
                loss_score = loss.mean().item()
                current = batch_number * batch_size + len(X)
                print(f' Average Loss: {loss_score}, Progress: {current}/{data_size}')
            #section-end
        #section-end
        #section-start place model in eval mode
        model.apply(reset_perturbation())
        model.eval()
        #section-end
#section-end
def test_epoch(testing_model, loss_function, testing_dataloader, batch_size, *, sample_testing_cutoff=4000): #section-start
    testing_model.eval()
    total_loss = torch.zeros(1)
    samples_tested = 0
    for batch_number, (X,_) in enumerate(testing_dataloader):
        loss = loss_function(X, testing_model(X))
        total_loss += loss.detach()
        samples_tested += batch_size
        if samples_tested >= sample_testing_cutoff:
            break
    average_loss = total_loss/samples_tested
    return average_loss.item()
#section-end
def egg_train_loop(*, #section-start
    model,
    loss_function,
    training_dataset,
    optimizer,
    batch_size,
    epochs=3,
    testing_dataset=None,
    training_epoch_granularity=7,
    training_epoch_dots=5,
    testing_epoch_dots=10):
    print('Training Starting!')
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        generator=torch.Generator(device=device),
        shuffle=True)
    if testing_dataset:
        testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
    training_loss_record = []
    testing_loss_record = []
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch} of {epochs}')
        egg_train_epoch(
            model=model,
            loss_function=loss_function,
            training_dataloader=training_dataloader,
            optimizer=optimizer,
            batch_size=batch_size,
            training_epoch_granularity=training_epoch_granularity,
            training_epoch_dots=training_epoch_dots)
        #TODO re-add #average_training_loss = test_epoch(model, loss_function, training_dataloader, batch_size)
        #TODO re-add #training_loss_record.append(average_training_loss)
        #print(f'Average Training Set Loss: {average_training_loss}')
#        if testing_dataset:
#            average_testing_loss = test_epoch(model, loss_function, testing_dataloader, batch_size)
#            testing_loss_record.append(average_testing_loss)
#            print(f'Average Test Set Loss: {average_testing_loss}')
    print('Training Complete!')
#  if testing_dataloader is not None:
#      return pd.DataFrame({'epoch':range(1,epochs+1), 'training loss':training_loss_record, 'testing loss':testing_loss_record})
#  else:
#      return pd.DataFrame({'epoch':range(1,epochs+1), 'training loss':training_loss_record})
#section-end
def VAE_loss_function(X, Y, ELBO): #section-start
    assert len(ELBO.shape) == 1
    return(-ELBO)
#section-end
some_data, _ = MNIST()
specific_image = some_data[0][0].unsqueeze(0).to(device)
def l1_loss_function_fixed(X, Y, Pred): #section-start
    assert specific_image.shape[1:]==Pred.shape[1:]
    return(torch.abs(specific_image-Pred).flatten(start_dim=1).sum(dim=1))
#section-end
def l1_loss_function_dynamic(X, Y, Pred): #section-start
    assert X.shape==Pred.shape
    return(torch.abs(X-Pred).flatten(start_dim=1).sum(dim=1))
#section-end
def l2_loss_function_dynamic(X, Y, Pred): #section-start
    assert X.shape==Pred.shape
    stdev = torch.ones_like(X)
    return(-diagonal_gaussian_unnormalized_log_likelyhood(
        mean=Pred,
        stdev=stdev,
        draw=X,
    ))
#section-end
def diagonal_gaussian_unnormalized_log_likelyhood(*, #section-start
    #section-start args
    mean,
    stdev,
    draw):
    #section-end
    #section-start """
    """
    I had to modify this one to return a vector of losses instead of a single unified loss. This makes it eggroll compatible.
    
    This just gets the log likelyhood of drawing something from a multivaritate gaussian with diagonal covariance matrix
    """
    #section-end
    #section-start validate input
    batch_size = mean.shape[0]
    assert stdev.shape[0] == batch_size
    #section-start assert stdev has no ambiguity when broadcast to mean size
    assert len(mean.shape) == len(stdev.shape)
    #section-end
    assert mean.shape == draw.shape
    assert torch.all(stdev > 0)
    #section-end
    #section-start flatten the inputs
    mean = torch.flatten(
        input=mean,
        start_dim=1)
    stdev = torch.flatten(
        input=stdev,
        start_dim=1)
    draw = torch.flatten(
        input=draw,
        start_dim=1)
    #section-end
    #section-start do the calculation!
    #section-start determine the probability when normalized
    identity_prob = (-0.5*(((mean-draw)/stdev)**2)).sum(dim=1)
    #section-end
    #section-start determine the determinant
    determinant_regularizer = torch.log(stdev).sum(dim=1)
    #section-end
    #section-start combine these together
    retval = identity_prob - determinant_regularizer
    #section-end
    #section-end
    #section-start validate output
    assert retval.shape[0] == batch_size
    assert len(retval.shape) == 1
    #section-end
    #section-start return output!
    return retval
    #section-end
#section-end
def generation_figure(decoder): #section-start
    def random_decoder_sample():
      return decoder(torch.zeros(1).unsqueeze(0))[0] #TODO remove
      return decoder(torch.randn(10).unsqueeze(0))[0]
    rows, columns, scaleup = 2, 8, 2
    figure = plt.figure(figsize=(columns* scaleup, rows*scaleup), layout='constrained')
    figure.suptitle('Random Draws p(~N(0,I))')
    figure.set_facecolor('lightyellow')
    figure.set_edgecolor('tan')
    figure.set_linewidth(1)
    axs = figure.subplots(nrows=rows, ncols=columns)
    for ax in axs.ravel():
        image = random_decoder_sample().detach().cpu()
        ax.imshow(image.squeeze(), cmap='grey', vmin=0, vmax=1)
        ax.set_title('')
        ax.set_axis_off()
    return figure
#section-end
def main(): #section-start
    #section-start set training parameters
    batch_size = 96
    #model = EggVAEGaussian(
    #    data_shape=torch.Size([1, 28, 28]),
    #    embedding_shape=torch.Size([10]),
    #    network_width=4
    #)
    model = EggSimpleNet(
        input_shape=torch.Size([1]),
        output_shape=torch.Size([1,28,28]),
        network_width=16)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-10, weight_decay=1e-2)
    loss_function = l2_loss_function_dynamic
    training_dataset, _ = MNIST()
    epochs=10
    #section-end
    #section-start run the train loop
    egg_train_loop(
        model=model,
        loss_function=loss_function,
        training_dataset=training_dataset,
        optimizer=optimizer,
        batch_size=batch_size,
        epochs=epochs)
    #section-end
    #section-start call the figure generator
    print("making generation figure")
    generation_figure(model).savefig("/app/figures/genfig.png")
    print("generation figure generated")
    #section-end
#section-end
#section-start call main
if __name__ == "__main__":
    main()
#section-end

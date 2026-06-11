#section-start setup
#section-start import stuff
import torch
import EggModule as egg
#section-end
#section-start write constants
MINIMUM_STDEV = 0.1
#section-end
#section-end
#section-start models
class EggSimpleNet(torch.nn.Module): #section-start
    #section-start """
    """
    An Egg-compatible simple neural network with two (primary) layers!

    Taste the eggggggggggg
    """
    #section-end
    #section-start attributes
    __slots__ = ['input_shape', 'stack', 'output_shape']
    input_shape:    torch.Size
    stack:          torch.nn.Sequential
    output_shape:   torch.Size
    #section-end
    def __init__(self, *, #section-start
        #section-start args
        input_shape,
        output_shape,
        network_width=128):
        #section-end
        #section-start init superclass
        super().__init__()
        #section-end
        #section-start set shape attributes
        self.input_shape = input_shape
        self.output_shape = output_shape
        #section-end
        #section-start build network!
        self.stack = torch.nn.Sequential(
            #egg.EggBatchNorm1d(
            #    num_features=input_shape.numel()
            #),
            egg.EggAffine(
                num_input_features=input_shape.numel(),
                num_output_features=network_width
            ),
            #egg.EggBatchNorm1d(
            #    num_features=network_width
            #),
            torch.nn.ReLU(),
            egg.EggAffine(
                num_input_features=network_width,
                num_output_features=network_width
            ),
            #egg.EggBatchNorm1d(
            #    num_features=network_width
            #),
            torch.nn.ReLU(),
            egg.EggAffine(
                num_input_features=network_width,
                num_output_features=output_shape.numel()
            )
        )
        #section-end
    #section-end
    def forward(self, x): #section-start
        #section-start validate input
        batch_size = x.shape[0]
        assert x.shape[1:] == self.input_shape
        #section-end
        #section-start flatten
        x = torch.flatten(
            input=x,
            start_dim=1
        )
        #section-end
        #section-start network!
        x = self.stack(x)
        #section-end
        #section-start unflatten
        x = torch.unflatten(
            input=x,
            dim=1,
            sizes=self.output_shape
        )
        #section-end
        #section-start validate output
        assert x.shape[0] == batch_size
        assert x.shape[1:] == self.output_shape
        #section-end
        #section-start return output
        return(x)
        #section-end
    #section-end
#section-end
class EggVAEGaussian(torch.nn.Module): #section-start
    #section-start """
    """
    The classical gaussian variational autoencoder but Egg compatible. The prior of this is the standard normal of whatever embedding shape is specified.
    """
    #section-end
    #section-start attributes
    #section-start slots
    __slots__ = [
        'data_shape',
        'embedding_features',
        'encoder',
        'decoder',
        'encoder_stdev_module',
        'decoder_stdev_module'
    ]
    #section-end
    #section-start typings
    data_shape:             torch.Size
    embedding_shape:        torch.Size
    encoder:                EggSimpleNet
    decoder:                EggSimpleNet
    encoder_stdev_module:   egg.EggVector
    decoder_stdev_module:   egg.EggVector
    #section-end
    #section-end
    def __init__(self, *, #section-start
        #section-start args
        data_shape=torch.Size([1, 28, 28]),
        embedding_shape=torch.Size([10]),
        network_width=128):
        #section-end
        #section-start init superclass
        super().__init__()
        #section-end
        #section-start body
        #section-start write shapes
        self.data_shape = data_shape
        self.embedding_shape = embedding_shape
        #section-end
        #section-start make networks
        self.encoder = EggSimpleNet(
            input_shape=data_shape,
            output_shape=embedding_shape,
            network_width=network_width)
        self.decoder = EggSimpleNet(
            input_shape=embedding_shape,
            output_shape=data_shape,
            network_width=network_width)
        #section-end
        #section-start set stdev modules
        self.encoder_stdev_module = egg.EggVector(
            num_features=1
        )
        self.decoder_stdev_module = egg.EggVector(
            num_features=1
        )
        #section-end
        #section-end
    #section-end
    def forward(self, data): #section-start
        #section-start validate input
        batch_size = data.shape[0]
        assert data.shape[1:] == self.data_shape
        #section-end
        #section-start run the models
        #section-start run the encoder!
        embedding_mean = self.encoder(data)
        embedding_stdev = torch.abs(self.encoder_stdev_module(batch_size=batch_size)) + MINIMUM_STDEV
        #section-start unfold stdev to match dimension number
        embedding_stdev_unfolded = torch.unflatten(
            input=embedding_stdev,
            dim=1,
            sizes=torch.Size([1]*len(self.embedding_shape))
        )
        #section-end
        embedding_sample = torch.normal(
            mean = embedding_mean,
            std = embedding_stdev_unfolded
        )
        #section-end
        #section-start declare the prior
        prior_mean = torch.zeros(
            size=torch.Size([batch_size])+self.embedding_shape
        )
        prior_stdev = prior_mean + 1
        #section-end
        #section-start run the decoder!
        decoding_mean = self.decoder(embedding_sample)
        decoding_stdev = torch.abs(self.decoder_stdev_module(batch_size=batch_size)) + MINIMUM_STDEV
        #section-start unfold stdev to match dimension number
        decoding_stdev_unfolded = torch.unflatten(
            input=decoding_stdev,
            dim=1,
            sizes=torch.Size([1]*len(self.data_shape))
        )
        #section-end
        #section-end
        #section-end
        #section-start do come calculations
        #section-start compute the likelyhoods
        embedding_sample_encoder_log_likelyhood = ( #section-start
            diagonal_gaussian_unnormalized_log_likelyhood(
                mean=embedding_mean,
                stdev=embedding_stdev_unfolded,
                draw=embedding_sample
            )
        )
        #section-end
        embedding_sample_prior_log_likelyhood = ( #section-start
            diagonal_gaussian_unnormalized_log_likelyhood(
                mean=prior_mean,
                stdev=prior_stdev,
                draw=embedding_sample
            )
        )
        #section-end
        data_decoding_log_likelyhood = ( #section-start
            diagonal_gaussian_unnormalized_log_likelyhood(
                mean=decoding_mean,
                stdev=decoding_stdev_unfolded,
                draw=data
            )
        )
        #section-end
        #section-end
        #section-start calculate the elbo
        evidence_lower_bound = (
            data_decoding_log_likelyhood +
            embedding_sample_prior_log_likelyhood -
            embedding_sample_encoder_log_likelyhood
        )
        #section-end
        #section-end
        #section-start validate output
        assert evidence_lower_bound.shape[0] == batch_size
        assert len(evidence_lower_bound.shape) == 1
        #section-end
        #section-start return the output (elbo)
        return(evidence_lower_bound)
        #section-end
    #section-end
#section-end
#section-end
#section-start helper functions
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
#section-end

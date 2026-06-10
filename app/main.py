#section-start import stuff
import torch
import EggModule as egg
#section-end
class EggSimpleNet(nn.Module): #section-start
    #section-start """
    """
    An Egg-compatible simple neural network with two (primary) layers!

    Taste the eggggggggggg
    """
    #section-end
    #section-start attributes
    __slots__ == ['input_shape', 'stack', 'output_shape']
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
        #section-start set shape attributes
        self.input_shape = input_shape
        self.output_shape = output_shape
        #section-end
        #section-start build network!
        self.stack = torch.nn.Sequential(
            egg.EggBatchNorm1d(
                num_features=input_shape.numel()
            ),
            egg.EggAffine(
                num_input_features=input_shape.numel(),
                num_output_features=network_width
            ),
            egg.EggBatchNorm1d(
                num_features=network_width
            ),
            torch.nn.ReLU(),
            egg.EggAffine(
                num_input_features=network_width,
                num_output_features=network_width
            ),
            egg.EggBatchNorm1d(
                num_features=network_width
            ),
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
class EggVAEGaussian(nn.Module):
    #section-start """
    """
    The classical gaussian variational autoencoder but Egg compatible. The prior of this is the standard normal of whatever embedding shape is specified.
    """
    #section-end
    #section-start attributes
    #section-start slots
    __slots__ == [
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
    def forward(self, data):

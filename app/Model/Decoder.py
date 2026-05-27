import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)
logger.info('Loading Decoder')
class Decoder(nn.Module):
  def __init__(self, *, embedding_dimensionality=10, data_shape=torch.Size([1, 28, 28]), layer_dimensionality=128, uniform_stdev=0.05):
    super().__init__()
    self.uniform_stdev = uniform_stdev
    self.dstack = nn.Sequential(
        nn.BatchNorm1d(embedding_dimensionality),
        nn.Linear(embedding_dimensionality, layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, data_shape.numel())
    )
    self.unflatten = nn.Unflatten(1, data_shape)
  def forward(self, input):
    mean = self.unflatten(self.dstack(input))
    stdev = self.uniform_stdev + torch.zeros_like(mean)
    return mean, stdev
logger.info('Verifying Decoder integrity...')
decoder_settings = {'embedding_dimensionality':2, 'data_shape':torch.Size([5,5]), 'layer_dimensionality':16}
blank_decoder = Decoder(**decoder_settings)
try:
  dummy_data = torch.zeros(torch.Size([2])+torch.Size([decoder_settings['embedding_dimensionality']]))
  dummy_mean, dummy_stdev = blank_decoder(dummy_data)
  expected_output_size = torch.Size([2]) + decoder_settings['data_shape']
  if (dummy_mean.size() != expected_output_size):
    raise ValueError(f'Unexpected Decoder mean output size {dummy_mean.size()}, expected {expected_output_size}')
  if (dummy_stdev.size() != expected_output_size):
    raise ValueError(f'Unexpected Decoder stdev output size {dummy_stdev.size()}, expected {expected_output_size}')
except Exception as e:
  raise ValueError('Decoder Inoperable. Repair needed.') from e

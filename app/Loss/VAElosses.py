import torch
def diagonal_guassian_unnormalized_log_likelyhood(mean, stdev, draw):
  mean = torch.flatten(mean)
  stdev = torch.flatten(stdev)
  draw = torch.flatten(draw)
  identity_prob = (-0.5*(((mean-draw)/stdev)**2)).sum()
  determinant_regularizer = torch.log(stdev).sum()
  return identity_prob - determinant_regularizer
def VAE_loss(model_input, model_output):
  encoder_mean = model_output['encoder_mean']
  encoder_stdev = model_output['encoder_stdev']
  encoder_draw = model_output['encoder_draw']
  decoder_mean = model_output['decoder_mean']
  decoder_stdev = model_output['decoder_stdev']
  dgull = diagonal_guassian_unnormalized_log_likelyhood
  reconstruction_term = dgull(decoder_mean, decoder_stdev, model_input)
  kl_term = dgull(torch.zeros_like(encoder_mean), torch.ones_like(encoder_mean), encoder_mean) - dgull(encoder_mean, encoder_stdev, encoder_draw)
  return -reconstruction_term -kl_term

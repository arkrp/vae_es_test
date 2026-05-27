import torch
import pandas as pd
from torch.utils.data import DataLoader
from math import ceil
"""
A simple generalizable training loop
"""
def train_epoch(training_model, loss_function, training_dataloader, optimizer, batch_size, *, training_epoch_granularity=7, training_epoch_dots=5):
  training_model.train()
  data_size = len(training_dataloader.dataset)
  num_batches = ceil(data_size/batch_size)
  batch_report_point = num_batches//training_epoch_granularity
  batch_dot_point = (batch_report_point//training_epoch_dots)
  for batch_number, (X,_) in enumerate(training_dataloader):
    loss = loss_function(X, training_model(X))
    loss = loss/batch_size # Normalize for batch size.
    if loss.isnan():
      raise RuntimeError(f'Nan loss occurred during training. D: {batch_number=}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    report_point = batch_number % batch_report_point
    dot_point = report_point % batch_dot_point
    if batch_dot_point - dot_point == 1:
      print('.', end='', flush=True)
    if batch_report_point - report_point == 1:
      loss_score = loss.item()
      current = batch_number * batch_size + len(X)
      print(f' Average Loss: {loss_score}, Progress: {current}/{data_size}')
def test_epoch(testing_model, loss_function, testing_dataloader, batch_size, *, sample_testing_cutoff=4000):
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
def train_loop(model, loss_function, training_dataset, optimizer, *, testing_dataset=None, batch_size=64, epochs=3, training_epoch_granularity=7, training_epoch_dots=5, testing_epoch_dots=10):
  print('Training Starting!')
  training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
  testing_dataloader = None
  training_loss_record = []
  testing_loss_record = []
  if testing_dataset:
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
  for epoch in range(1, epochs+1):
    print(f'Epoch {epoch} of {epochs}')
    train_epoch(model, loss_function, training_dataloader, optimizer, batch_size, training_epoch_granularity=training_epoch_granularity, training_epoch_dots=training_epoch_dots)
    average_training_loss = test_epoch(model, loss_function, training_dataloader, batch_size)
    training_loss_record.append(average_training_loss)
    print(f'Average Training Set Loss: {average_training_loss}')
    if testing_dataset:
      average_testing_loss = test_epoch(model, loss_function, testing_dataloader, batch_size)
      testing_loss_record.append(average_testing_loss)
      print(f'Average Test Set Loss: {average_testing_loss}')
  print('Training Complete!')
  if testing_dataset:
      return pd.DataFrame({'epoch':range(1,epochs+1), 'training loss':training_loss_record, 'testing loss':testing_loss_record})
  else:
      return pd.DataFrame({'epoch':range(1,epochs+1), 'training loss':training_loss_record})

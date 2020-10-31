'''
Simple example of AMP Automatic Mixed Precision in PyTorch 1.6 using
FizzBuzz, inspired from the excellent blog post
https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
PyTorch implementation borrowed heavily from
https://github.com/luckytoilet/fizz-buzz-pytorch

TensorBoard
In another window, run  tensorboard --logdir=runs
    browse to http://localhost:6006/

Run nvidia-smi for GPU info
'''
import gc
import sys
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from colorama import init as colorama_init, deinit, Fore

NUM_DIGITS = 14
NUM_HIDDEN = 100
# With NUM_DIGITS 10, there are only 923 (1024-101) items in the training data
# so don't set the batch size really high or you'll skip a lot
# of data.  At 256, 155 of the 923 items will be missed, at
# 128, 27 will be
#BATCH_SIZE = 128
BATCH_SIZE = 512
OUTPUT_EPOCH_GAP = 100
LEARNING_RATE = 0.05
EPOCHS = 10000


class FizzBuzz():
    def __init__(self, num_digits):
        self.num_digits = num_digits

    # Represent each input by an array of its binary digits (1024 2**10)
    def binary_encode(self, i):
        return np.array([i >> d & 1 for d in range(self.num_digits)])

    def binary_encode_as_numpy_array(self, start, end=None):
        end = 2 ** self.num_digits if end is None else end
        return [self.binary_encode(i) for i in range(start, end)]

    def test_data_start(self):
        return ((self.num_digits - 9) * 100) + 1 

    # One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
    def encode(self, i):
        if   i % 15 == 0: return 3
        elif i % 5  == 0: return 2
        elif i % 3  == 0: return 1
        else:             return 0

    def decode(self, i, prediction):
        color = self.color(self.match(i,prediction))
        return color + [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

    def match(self, i, prediction):
        return self.encode(i) == prediction

    def color(self, match):
        return Fore.GREEN if match else Fore.RED


class FizzBuzzNetwork(nn.Module):
    def __init__(self):
        super(FizzBuzzNetwork, self).__init__()
        self.main = nn.Sequential(
            torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(NUM_HIDDEN, 4)
        )

    def forward(self, input):
        output = self.main(input)
        return output


class FizzBuzzDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, batch_size):
        self.xs = x
        self.ys = y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.xs) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        X = self.xs[start:end]
        Y = self.ys[start:end]
        return X, Y

def print_model_params(model):
    for name, param in model.named_parameters():
        #gradients = param.grad
        #github.com/pytorch/pytorch/pull/30531
        print( f'Param Name: {name} Value: {param.data} Gradient: {param.grad}')

def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    return time.time()

def end_timer(start_time):
    torch.cuda.synchronize()
    return ((time.time()-start_time), torch.cuda.max_memory_allocated())

def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA required") 
    device = torch.device("cuda")
    print(f'Pytorch {torch.__version__} Device {device}')

    colorama_init()
    random.seed() # OS provides seed

    fizz_buzz = FizzBuzz(NUM_DIGITS)
    test_data_start = fizz_buzz.test_data_start()
    print(f'Fizz Buzz from 1 to {test_data_start-1} '
        f'One Hot Input {NUM_DIGITS} ({2**NUM_DIGITS}) '
        f'Hidden Layer {NUM_HIDDEN} '
        f'Batch Size {BATCH_SIZE}')

    writer = SummaryWriter()

    # Train on 101 - 2**NUM_DIGITS (1024,2048,...)
    train_x_np = fizz_buzz.binary_encode_as_numpy_array(test_data_start)

    train_x = torch.torch.cuda.FloatTensor(train_x_np, device=device)
    train_x.requires_grad_()

    # Loss function has something to do with using a Long here
    # Compute correct results for 101 - 1024, the "labels"
    train_y_array = [fizz_buzz.encode(i) for i in range(test_data_start, 2 ** NUM_DIGITS)]

    train_y = torch.cuda.LongTensor(train_y_array, device=device)

    training_set = FizzBuzzDataset(train_x, train_y, BATCH_SIZE)

    # Define the model
    model = FizzBuzzNetwork().to(device)
    # Stochastic Gradient Descent (mini-batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) 
    loss_fn = torch.nn.CrossEntropyLoss()


    # print_model_params(model)

    # AMP initialization
    scaler = torch.cuda.amp.GradScaler()

    # Start training
    start_time = start_timer()
    for epoch in range(EPOCHS):
        max_loss = 0.0
        min_loss = sys.float_info.max

        # Since we are slicing BATCH_SIZE arrays from an already created
        # Float Tensor, the PyTorch DataLoader can't do our batching, even
        # if we set the Batch Size to 1 and try to return our own batch data.
        # DataLoader provides a lot of features but isn't very flexible.
        # So we do our own batch and our own random sequence of batch
        # starting indexes
        batch_list = list(range(0, len(training_set)))
        random.shuffle(batch_list)
        for index in batch_list:
            (batchX, batchY) = training_set[index]

            optimizer.zero_grad()

            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                y_batch_pred = model(batchX)
                #print_model_params(model)
                loss = loss_fn(y_batch_pred, batchY)

            scaler.scale(loss).backward()
            
            loss_val = loss.item()
            min_loss = min(loss_val, min_loss)
            max_loss = max(loss_val, max_loss)

            scaler.step(optimizer)
            scaler.update()

        # Find loss on all of the training data
        with torch.cuda.amp.autocast():
            y_pred_all = model(train_x)
            loss_all = loss_fn(y_pred_all, train_y)

        loss_all_val = loss_all.item()

        # Without AMP
        # loss_all_val = loss_fn(model(train_x), train_y)
        # loss_val = loss_all_val.item()

        min_max_range = max_loss - min_loss

        writer.add_scalar("Loss/train", loss_all_val, epoch)
        writer.flush()

        if epoch % OUTPUT_EPOCH_GAP == 0:
            print (f'Epoch: {epoch:>7} '
                f'Loss: {loss_all_val:.18f} '
                f'Min: {min_loss:.18f} '
                f'Max: {max_loss:.18f} '
                f'Range: {min_max_range:.5f}')

    writer.close()
    time_taken, max_memory_allocated = end_timer(start_time)

    print(f'Epochs {EPOCHS:,} Duration {time_taken:.3f} Max Tensor Memory {max_memory_allocated:,}')

    # Test on 1-100 ( or test_data_start )
    test_x_numpy = fizz_buzz.binary_encode_as_numpy_array(1, test_data_start)
    test_x = torch.cuda.FloatTensor(test_x_numpy).to(device)
    test_y = model(test_x)
    predictions = zip(range(1, test_data_start), list(test_y.max(1)[1].data.tolist()))

    matches = 0
    for(i, x) in predictions:
        match = fizz_buzz.match(i, x)
        matches = matches + match
        print(fizz_buzz.color(match) + fizz_buzz.decode(i, x), end=", ")
    print()
    print( f'{Fore.RESET} Incorrect: {test_data_start-1-matches} Correct: {matches}' )

if __name__ == "__main__":
    main()

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections.abc import Callable
from pathlib import Path
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib.request
from rng import RandomNumberGenerator

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
CONTEXT_LEN = 25
EPOCH_LEN = 100
USER_INPUT_FILE = Path( 'userinput.txt' )

class NeuralNetwork( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( 10*CONTEXT_LEN, 256 ),
            nn.ReLU(),
            nn.Linear( 256, 256 ),
            nn.ReLU(),
            nn.Linear( 256, 10 ),
        )

    def forward( self, x ):
        x = self.flatten( x )
        logits = self.linear_relu_stack( x )
        return logits

def train( dataloader: DataLoader, model: NeuralNetwork, loss_fn: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ], optimizer: torch.optim.Optimizer ):
    size = len( dataloader.dataset )
    model.train()
    for batch, (X, y) in enumerate( dataloader ):
        X, y = X.to( device ), y.to( device )

        # Compute prediction error
        pred: torch.Tensor = model( X )
        loss = loss_fn( pred, y )

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), ( batch + 1 ) * len( X )
            print( f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]" )


if __name__ == '__main__':
    print( "Initializing..." )

    # Try to load model from file, otherwise make it from scratch
    model_path = Path( "model.pth" )
    model = NeuralNetwork().to( device )
    if model_path.exists():
        print( f"Loading pretrained model from {model_path}" )
        model.load_state_dict( torch.load( model_path, weights_only=True ) )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( model.parameters(), lr=1e-2 )
    prediction_accuracies = np.empty( (1, EPOCH_LEN) )


    # help message dict
    help_message_dict = dict(
        help='print this help message',
        q='close the program',
        w='write current model to file',
        l='reload model from file',
        s=f'display accuracy of model in epochs of {EPOCH_LEN}',
        r='get a truly random number from random.org',
        repoch=f'get {EPOCH_LEN} (EPOCH_LEN) truly random numbers from random.org',
        read=f'read user inputs from {USER_INPUT_FILE} and return control when finished'
    )
    max_key_length = max( [ len( k ) for k in help_message_dict.keys() ] )
    max_val_length = max( [ len( v ) for v in help_message_dict.values() ] )
    help_format = '{0:>%d}: {1:>%d}' % (max_key_length, max_val_length)
    rng = RandomNumberGenerator()

    def help_fn( help_message_dict, help_format, **kwargs ):
        for k, v in help_message_dict.items():
            print( help_format.format( k, v ) )
    def save_fn( model, model_path, **kwargs ):
        print( f"Saving model to {model_path}..." )
        torch.save( model.state_dict(), model_path )
        print( f"Save complete!" )
    def load_fn( model, model_path, **kwargs ):
        print( f"Loading model from {model_path}..." )
        model.load_state_dict( torch.load( model_path, weights_only=True ) )
        print( f"Loading complete!" )
    def show_fn( prediction_accuracies, **kwargs ):
        prediction_accuracies_full = prediction_accuracies[ :-1, : ]
        #plt.xlabel( 'Percentage (%) given by the model to the user\'s answer' )
        #plt.ylabel( 'Number of predictions' )
        #cmap = cm.get_cmap( 'hsv', prediction_accuracies.shape[ 0 ] ) # 1 more than plotted b/c first and last are same color
        #for i in range( prediction_accuracies_full.shape[ 0 ] ):
        #    plt.hist( prediction_accuracies_full[ i ], bins=10, range=(0, 40), label=f'epoch {i}', histtype='step', density=True, color=cmap( i ) )
        #    plt.vlines( np.mean( prediction_accuracies_full[ i ] ), 0, 1, label=f'epoch {i} mean', colors=cmap( i ) )
        plt.xlabel( 'Epoch' )
        plt.ylabel( 'Percentage (%) given by the model to the user\'s answer' )
        epochs = np.arange( 0, prediction_accuracies_full.shape[ 0 ], 1 )
        plt.errorbar( epochs, np.mean( prediction_accuracies_full, axis=1 ), np.std( prediction_accuracies_full, axis=1 ) )
        plt.legend()
        plt.show()
    def rand_fn( rng, **kwargs ):
        return rng.next()
    def rand_epoch_fn( rng, **kwargs ):
        return rng.next( EPOCH_LEN )
    def read_fn( **kwargs ):
        with open( USER_INPUT_FILE ) as file:
            filedata = file.read()
        return np.array( filedata.splitlines(), dtype=int )
        

    # lambdas for all special commands
    special_commands = dict(
        help=help_fn,
        q=lambda **kwargs : ( exit( 0 ) ),
        w=save_fn,
        l=load_fn,
        s=show_fn,
        r=rand_fn,
        repoch=rand_epoch_fn,
        read=read_fn
    )

    # Main program loop
    print( "Please enter a number 0-9, type help for a list of commands, or q to close" )
    user_input = ""
    prev_user_inputs = deque( maxlen=CONTEXT_LEN )
    iter = 0
    auto_user_inputs = np.empty( (0) )
    while True:
        try:
            # Iterate through auto_user_inputs if it exists, otherwise prompt the user for a value
            if auto_user_inputs.shape[ 0 ] == 0:
                user_input = input( "> " )
                user_input = user_input.lower().strip()
            else:
                user_input = auto_user_inputs[ 0 ]
                auto_user_inputs = auto_user_inputs[ 1: ]

            next_loop = False
            for k in special_commands.keys():
                if user_input == k:
                    val = special_commands[ k ]( model=model, 
                                                 model_path=model_path, 
                                                 help_message_dict=help_message_dict,
                                                 help_format=help_format,
                                                 prediction_accuracies=prediction_accuracies,
                                                 rng=rng )
                    if k == 'r':
                        user_input = val[ 0 ]
                    elif k == 'repoch' or k == 'read':
                        auto_user_inputs = val[ 1: ]
                        user_input = val[ 0 ]
                    else: next_loop = True
            if next_loop:
                continue

            user_input = int( user_input )
            if ( user_input > 9 ) or ( user_input < 0 ):
                raise ValueError( f"Out of bounds value {user_input}" )

            if len( prev_user_inputs ) < CONTEXT_LEN:
                print( f'Input {user_input} received and added to context ({len( prev_user_inputs )}/{CONTEXT_LEN})' )
                prev_user_inputs.append( user_input )
                continue

            # Sample the model and show the user
            input_tensor = torch.zeros( 1, CONTEXT_LEN, 10, dtype=torch.float )
            for i in range( CONTEXT_LEN ):
                input_tensor[ :, i, prev_user_inputs[ i ] ] = 1
            input_tensor = input_tensor.to( device )

            logits = model( input_tensor )
            pred_probs: torch.Tensor = nn.Softmax( dim=1 )( logits )
            y_pred = pred_probs.argmax( 1 )[ 0 ].item()
            y_prob = pred_probs[ :, y_pred ].item()

            #print( f"Model predicted {y_pred} with probability {y_prob*100:.1f}%" )
            print( f"Model predicted {user_input} with probability {pred_probs[ :, user_input ].item()*100:.1f}%" )

            # Train the model on the actual user input
            print( f"Training model..." )
            output_tensor = torch.zeros( 1, 10, dtype=torch.float )
            output_tensor[ :, user_input ] = 1
            output_tensor = output_tensor.to( device )

            dataset = TensorDataset( input_tensor, output_tensor )
            dataloader = DataLoader( dataset )
            train( dataloader, model, loss_fn, optimizer )
            print( f"Trained!" )

            # Record the accuracy of the model for showing the user
            within_epoch_iter = iter % EPOCH_LEN
            prediction_accuracies[ -1, within_epoch_iter ] = pred_probs[ :, user_input ].item()*100
            print( f"Iteration {within_epoch_iter}/{EPOCH_LEN} for epoch {prediction_accuracies.shape[ 0 ]-1}" )

            iter += 1
            if iter % EPOCH_LEN == 0:
                prediction_accuracies = np.vstack( (prediction_accuracies, np.empty( (1, EPOCH_LEN) )) )

            prev_user_inputs.append( user_input )

        except EOFError:
            print( "EOF, exiting..." )
            exit( 0 )
        except KeyboardInterrupt:
            print( "Keyboard Interrupt, exiting...")
            exit( 0 )
        except ValueError:
            print( "Please enter a number 0-9" )
            continue

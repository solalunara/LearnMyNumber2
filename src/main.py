import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections.abc import Callable
from pathlib import Path
import numpy as np
from collections import deque

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
CONTEXT_LEN = 25

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


    # help message dict
    help_message_dict = dict(
        help='print this help message',
        q='close the program',
        w='write current model to file',
        l='reload model from file'
    )
    max_key_length = max( [ len( k ) for k in help_message_dict.keys() ] )
    max_val_length = max( [ len( v ) for v in help_message_dict.values() ] )
    help_format = '{0:>%d}: {1:>%d}' % (max_key_length, max_val_length)

    def help_fn( help_message_dict=help_message_dict, help_format=help_format ):
        for k, v in help_message_dict.items():
            print( help_format.format( k, v ) )
    def save_fn( model=model, model_path=model_path ):
        print( f"Saving model to {model_path}..." )
        torch.save( model.state_dict(), model_path )
        print( f"Save complete!" )
    def load_fn( model=model, model_path=model_path ):
        print( f"Loading model from {model_path}..." )
        model.load_state_dict( torch.load( model_path, weights_only=True ) )
        print( f"Loading complete!" )

    # lambdas for all special commands
    special_commands = dict(
        help=help_fn,
        q=lambda : ( exit( 0 ) ),
        w=save_fn,
        l=load_fn
    )

    # Main program loop
    print( "Please enter a number 0-9, type help for a list of commands, or q to close" )
    user_input = ""
    prev_user_inputs = deque( maxlen=CONTEXT_LEN )
    while True:
        try:
            user_input = input( "> " )
            user_input = user_input.lower().strip()

            next_loop = False
            for k in special_commands.keys():
                if user_input == k:
                    special_commands[ k ]()
                    next_loop = True
            if next_loop:
                continue

            user_input = int( user_input )
            if ( user_input > 9 ) or ( user_input < 0 ):
                raise ValueError( f"Out of bounds value {user_input}" )

            if len( prev_user_inputs ) < CONTEXT_LEN:
                print( f'Input {user_input} received and added to context ({len( prev_user_inputs )}/{CONTEXT_LEN})' )
                prev_user_inputs.append( user_input )
                continue

            prev_user_inputs.append( user_input )

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

            previous_user_input = user_input

        except EOFError:
            print( "EOF, exiting..." )
            exit( 0 )
        except KeyboardInterrupt:
            print( "Keyboard Interrupt, exiting...")
            exit( 0 )
        except ValueError:
            print( "Please enter a number 0-9" )
            continue

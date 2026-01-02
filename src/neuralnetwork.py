import torch.nn as nn
from pathlib import Path
import numpy as np
from rng import RandomNumberGenerator
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
from collections.abc import Callable

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork( nn.Module ):
    def __init__( self, model_path: Path, epoch_len: int = 300, context_len: int = 25, model_width: int = 32, user_input_file: Path | None = None, lr: float = 1e-3 ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.history = np.empty( 0, dtype=int )
        self.accuracy_history = np.empty( 0, dtype=float )
        self.rng = RandomNumberGenerator()

        self.model_path = model_path
        self.user_input_file = user_input_file
        self.epoch_len = epoch_len
        self.model_width = model_width
        self.context_len = context_len
        self.debug = True

        # help message dict
        self.help_message_dict = dict(
            help='print this help message',
            q='close the program',
            w='write current model to file',
            l='reload model from file',
            s=f'display accuracy of model in epochs of {self.epoch_len}',
            r='get a truly random number from random.org',
            repoch=f'get {self.epoch_len} (epoch length) truly random numbers from random.org',
            read=f'read user inputs from {self.user_input_file} and return control when finished'
        )
        max_key_length = max( [ len( k ) for k in self.help_message_dict.keys() ] )
        max_val_length = max( [ len( v ) for v in self.help_message_dict.values() ] )
        self.help_format = '{0:>%d}: {1:>%d}' % (max_key_length, max_val_length)

        # lambdas for all special commands
        self.special_commands = dict()
        for command in self.help_message_dict.keys():
            self.special_commands[ command ] = getattr( self, f'{command}_fn' )

        # Model
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( 10*self.context_len, self.model_width ),
            nn.ReLU(),
            nn.Linear( self.model_width, self.model_width ),
            nn.ReLU(),
            nn.Linear( self.model_width, self.model_width ),
            nn.ReLU(),
            nn.Linear( self.model_width, 10 ),
        )

        # Training parameters
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam( self.parameters(), lr=lr )

    def forward( self, x ):
        x = self.flatten( x )
        logits = self.linear_relu_stack( x )
        return logits

    def help_fn( self ):
        for k, v in self.help_message_dict.items():
            print( self.help_format.format( k, v ) )

    def q_fn( self ):
        exit( 0 )

    def w_fn( self ):
        print( f"Saving model to {self.model_path}..." )
        torch.save( self.state_dict(), self.model_path )
        print( f"Save complete!" )

    def l_fn( self ):
        print( f"Loading model from {self.model_path}..." )
        self.load_state_dict( torch.load( self.model_path, weights_only=True ) )
        print( f"Loading complete!" )

    def s_fn( self ):
        #plt.xlabel( 'Percentage (%) given by the model to the user\'s answer' )
        #plt.ylabel( 'Number of predictions' )
        #cmap = cm.get_cmap( 'hsv', prediction_accuracies.shape[ 0 ] ) # 1 more than plotted b/c first and last are same color
        #for i in range( prediction_accuracies_full.shape[ 0 ] ):
        #    plt.hist( prediction_accuracies_full[ i ], bins=10, range=(0, 40), label=f'epoch {i}', histtype='step', density=True, color=cmap( i ) )
        #    plt.vlines( np.mean( prediction_accuracies_full[ i ] ), 0, 1, label=f'epoch {i} mean', colors=cmap( i ) )
        #plt.legend()
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Percentage (%) given by the model to the user\'s answer' )
        n_epochs = len( self.accuracy_history ) // self.epoch_len
        epochs = np.arange( 0, n_epochs, 1 )
        accuracy_epochs = self.accuracy_history[ :n_epochs * self.epoch_len ].reshape( -1, self.epoch_len )
        plt.errorbar( epochs, np.mean( accuracy_epochs, axis=1 ), np.std( accuracy_epochs, axis=1 ) )
        plt.show()

    def r_fn( self ):
        self.training_loop( np.array( self.rng.next(), dtype=int ) )

    def repoch_fn( self ):
        self.training_loop( np.array( self.rng.next( self.epoch_len ), dtype=int ) )

    def read_fn( self ):
        if self.user_input_file is not None:
            with open( self.user_input_file ) as file:
                filedata = file.read()
            self.training_loop( np.array( filedata.splitlines(), dtype=int ) )
        else:
            print( 'User input file not set - please set it for the model' )
    
    def training_loop( self, input: np.ndarray | str ):
        try:
            if isinstance( input, str ):
                # Command handling sub-block
                for k in self.special_commands.keys():
                    if input == k:
                        self.special_commands[ k ]()
                        return # do no more processing for commands

                # We also get a string for raw user input from the console, so try to convert to an int
                input = int( input )
                if ( input > 9 ) or ( input < 0 ):
                    raise ValueError( f"Out of bounds value {input}" )
                input = np.array( [ input ], dtype=int )
            # We can now safely assume input is a well-formed array
            input_len = input.shape[ 0 ]

            # Add values to context until it's full, and return if we don't have enough
            self.history = np.append( self.history, input )
            if len( self.history ) < self.context_len:
                print( f'Context {len( self.history )} / {self.context_len}' )
                return

            # Sample the model, with input_len inputs, each one with its own history including the previous elements in input
            model_input_tensor = torch.zeros( input_len, self.context_len, 10, dtype=torch.float )
            history_start_index = len( self.history ) - input_len - self.context_len
            for i in range( input_len ):
                for j in range( self.context_len ):
                    model_input_tensor[ i, j, self.history[ history_start_index + i + j ] ] = 1
            model_input_tensor = model_input_tensor.to( device ) # send to gpu after all memory modifications done, if we are sending to the gpu

            logits = self( model_input_tensor )
            pred_probs: torch.Tensor = nn.Softmax( dim=1 )( logits )
            pred_probs.detach() #to convert to scalars
            y_preds = pred_probs.argmax( 1 )
            y_probs = np.empty( len( pred_probs ) )
            actual_probs = np.empty( len( pred_probs ) )
            for i in range( len( y_preds ) ):
                y_probs[ i ] = pred_probs[ i, y_preds[ i ] ].item()
                actual_probs[ i ] = pred_probs[ i, input[ i ] ].item()

            for i in range( len( y_probs ) ):
                print( f"Model predicted {y_preds[ i ]} with probability {y_probs[ i ]*100:.1f}% - actual {input[ i ]} (model probability {actual_probs[ i ]*100:.1f}%)" )

            # Train the model on the actual user input
            print( f"Training model..." )
            output_tensor = torch.tensor( input, dtype=torch.long ).to( device )

            dataset = TensorDataset( model_input_tensor, output_tensor )
            dataloader = DataLoader( dataset, batch_size=input_len )
            self.train_on_dataloader( dataloader, self.loss_fn, self.optimizer )
            print( f"Trained!" )

            # Record the accuracy of the model for showing the user
            self.accuracy_history = np.append( self.accuracy_history, actual_probs )

        except EOFError:
            print( "EOF, exiting..." )
            exit( 0 )
        except KeyboardInterrupt:
            print( "Keyboard Interrupt, exiting...")
            exit( 0 )
        except ValueError:
            e = sys.exception()
            print( f"{sys.exception()} - (please make sure you enter a number 0-9)" )

    def train_on_dataloader( self, dataloader: DataLoader, loss_fn: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ], optimizer: torch.optim.Optimizer ):
        size = len( dataloader.dataset )
        self.train()
        for batch, (X, y) in enumerate( dataloader ):
            X, y = X.to( device ), y.to( device )
            optimizer.zero_grad()
            # Optional debug: capture parameter norm before the forward pass
            if getattr( self, 'debug', False ):
                param_norm_before = 0.0
                for p in self.parameters():
                    param_norm_before += p.data.norm().item()

            # Compute prediction error
            pred: torch.Tensor = self( X )
            loss = loss_fn( pred, y )

            # Backpropagation
            loss.backward()
            # Optional debug: print gradient norm to verify gradients are non-zero
            if getattr( self, 'debug', False ):
                grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm().item()
                print( f"[DEBUG] grad_norm={grad_norm:.6f}" )

            optimizer.step()

            # Optional debug: capture parameter norm after the optimizer step
            if getattr( self, 'debug', False ):
                param_norm_after = 0.0
                for p in self.parameters():
                    param_norm_after += p.data.norm().item()
                print( f"[DEBUG] param_norm_before={param_norm_before:.6f} param_norm_after={param_norm_after:.6f} delta={param_norm_after-param_norm_before:.6f}" )

            loss, current = loss.item(), ( batch + 1 ) * len( X )
            print( f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]" )
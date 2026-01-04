import torch.nn as nn
from pathlib import Path
import numpy as np
from rng import PseudoRandomNumberGenerator
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
from collections.abc import Callable
import pandas as pd

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork( nn.Module ):
    def __init__( self, context_len: int = 25, model_width: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4 ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.history = np.empty( 0, dtype=int )

        self.nan_dict = dict( train=np.full( (1), np.nan, dtype=float ),
                              test=np.full( (1), np.nan, dtype=float ),
                              random=np.full( (1), np.nan, dtype=float ) )

        self.loss_history = pd.DataFrame( self.nan_dict )

        self.model_width = model_width
        self.context_len = context_len

        # Model
        self.model = nn.Sequential(
            nn.Linear( 10*self.context_len, self.model_width ),
            nn.ReLU(),
            nn.Linear( self.model_width, self.model_width ),
            nn.ReLU(),
            nn.Linear( self.model_width, 10 ),
        )

        # Training parameters
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam( self.parameters(), lr=lr, weight_decay=weight_decay )

    def forward( self, x ):
        x = self.flatten( x )
        logits = self.model( x )
        return logits

    def eval_model( self, data: np.ndarray, data_type: str ):
        """
        Evaluates the model without training it. Still appends to model history.
        
        :param data: Numpy array containing the real data which the model will be evaluated against
        :type data: np.ndarray

        :param data_type: Type of data - one of 'train', 'test', or 'random'. This determines the loss
        history that is appended to, and if the data type is 'train' the model will be trained on the result
        :type data_type: str


        :returns torch.Tensor: the model input tensor passed to the model, containing staggered history
        information for each element up to a maximum length of self.context_len
        """
        self.eval()

        data_len = data.shape[ 0 ]

        # Add values to context until it's full, and return if we don't have enough
        self.history = np.append( self.history, data )
        if len( self.history ) < self.context_len:
            print( f'Context {len( self.history )} / {self.context_len}' )
            return

        # Sample the model, with input_len inputs, each one with its own history including the previous elements in input
        model_input_tensor = torch.zeros( data_len, self.context_len, 10, dtype=torch.float )
        history_start_index = len( self.history ) - data_len - self.context_len
        for i in range( data_len ):
            for j in range( self.context_len ):
                model_input_tensor[ i, j, self.history[ history_start_index + i + j ] ] = 1
        model_input_tensor = model_input_tensor.to( device ) # send to gpu after all memory modifications done, if we are sending to the gpu

        logits = self( model_input_tensor )
        pred_probs: torch.Tensor = nn.Softmax( dim=1 )( logits )
        pred_probs = pred_probs.detach() #to convert to scalars
        y_preds = pred_probs.argmax( 1 )
        y_probs = np.empty( len( pred_probs ) )
        actual_probs = np.empty( len( pred_probs ) )
        for i in range( len( y_preds ) ):
            y_probs[ i ] = pred_probs[ i, y_preds[ i ] ].item()
            actual_probs[ i ] = pred_probs[ i, data[ i ] ].item()

        #for i in range( len( y_probs ) ):
            #print( f"Model predicted {y_preds[ i ]} with probability {y_probs[ i ]*100:.1f}% - actual {data[ i ]} (model probability {actual_probs[ i ]*100:.1f}%)" )

        output_tensor = torch.tensor( data, dtype=torch.long ).to( device )
        loss: torch.Tensor = self.loss_fn( logits, output_tensor )
        loss_value = loss.item()

        # Append the loss to the array by changing the last non-nan value, or adding another row if necessary
        last_valid_index = self.loss_history[ data_type ].last_valid_index()
        if last_valid_index is None:
            last_valid_index = -1
        loss_index = last_valid_index + 1
        if loss_index >= self.loss_history.shape[ 0 ]:
            self.loss_history = pd.concat( [ self.loss_history, pd.DataFrame( self.nan_dict, index=[ loss_index ] ) ] )

        self.loss_history.loc[ loss_index, data_type ] = loss_value
        
        if data_type == 'train' or data_type == 'random':
            self.train()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_on_dataloader( self, dataloader: DataLoader, loss_fn: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ], optimizer: torch.optim.Optimizer ):
        size = len( dataloader.dataset )
        self.train()
        for batch, (X, y) in enumerate( dataloader ):
            X, y = X.to( device ), y.to( device )
            optimizer.zero_grad()

            # Compute prediction error
            pred: torch.Tensor = self( X )
            loss = loss_fn( pred, y )

            # Backpropagation
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), ( batch + 1 ) * len( X )
            #print( f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]" )



class InteractableNeuralNetwork( NeuralNetwork ):
    def __init__( self, model_path: Path, epoch_len: int = 300, context_len: int = 5, model_width: int = 5, user_input_file: Path | None = None, lr: float = 1e-3, should_train: bool = False, eval_ratio: float = 0.3, weight_decay: float = 1e-4 ):
        super().__init__( context_len=context_len, model_width=model_width, lr=lr, weight_decay=weight_decay )
        self.rng = PseudoRandomNumberGenerator()
        self.should_train = should_train
        self.eval_ratio = eval_ratio

        self.model_path = model_path
        self.user_input_file = user_input_file
        self.epoch_len = epoch_len

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
        plt.xlabel( 'Train/Eval call' )
        plt.ylabel( 'Perplexity' )
        for label, color in zip( self.loss_history.keys(), [ 'b', 'orange', 'm' ] ):
            loss_iters = np.arange( 0, self.loss_history[ label ].shape[ 0 ], 1 )
            plt.plot( loss_iters, np.exp( self.loss_history[ label ] ), label=label, color=color )
        plt.legend()
        plt.show()

    def r_fn( self ):
        self.eval_model( np.array( self.rng.next(), dtype=int ), 'random' )

    def repoch_fn( self ):
        self.eval_model( np.array( self.rng.next( self.epoch_len ), dtype=int ), 'random' )

    def read_fn( self ):
        if self.user_input_file is not None:
            with open( self.user_input_file ) as file:
                filedata = file.read()
            data = filedata.splitlines()

            # Ordering is important here to preserve context
            train_test_split_index = int( ( 1 - self.eval_ratio ) * len( data ) )
            train = data[ :train_test_split_index ]
            test = data[ train_test_split_index: ]
            self.eval_model( np.array( train, dtype=int ), 'train' )
            self.eval_model( np.array( test, dtype=int ), 'test' )
        else:
            print( 'User input file not set - please set it for the model' )

    def exec_command( self, input: str ):
        """
        Execute the 'input' command.

        :param input: The command as a string to execute
        :type input: str
        """
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
        except ValueError:
            print( f'{sys.exception()} - please make sure you enter a number 0-9 or a valid command' )
            return

        # input is now a numpy array of dtype=int
        self.eval_model( input, 'train' if self.should_train else 'test' )

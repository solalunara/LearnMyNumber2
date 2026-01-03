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
    def __init__( self, context_len: int = 25, model_width: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4 ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.history = np.empty( 0, dtype=int )
        self.train_accuracy_history = np.empty( 0, dtype=float )
        self.test_accuracy_history = np.empty( 0, dtype=float )

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

    def eval_model( self, test_data: np.ndarray, is_train_data: bool = False ):
        """
        Evaluates the model without training it. Still appends to model history.
        
        :param test_data: Numpy array containing the real data which the model will be evaluated against
        :type test_data: np.ndarray

        :param is_train_data: Whether or not the passed data is also training data. This is used to choose
        which accuracy history the results should be appended to, train or test.

        :type save_to_accuracy_history: bool

        :returns torch.Tensor: the model input tensor passed to the model, containing staggered history
        information for each element up to a maximum length of self.context_len
        """
        self.eval()

        test_len = test_data.shape[ 0 ]

        # Add values to context until it's full, and return if we don't have enough
        self.history = np.append( self.history, test_data )
        if len( self.history ) < self.context_len:
            print( f'Context {len( self.history )} / {self.context_len}' )
            return

        # Sample the model, with input_len inputs, each one with its own history including the previous elements in input
        model_input_tensor = torch.zeros( test_len, self.context_len, 10, dtype=torch.float )
        history_start_index = len( self.history ) - test_len - self.context_len
        for i in range( test_len ):
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
            actual_probs[ i ] = pred_probs[ i, test_data[ i ] ].item()

        for i in range( len( y_probs ) ):
            print( f"Model predicted {y_preds[ i ]} with probability {y_probs[ i ]*100:.1f}% - actual {test_data[ i ]} (model probability {actual_probs[ i ]*100:.1f}%)" )

        if is_train_data:
            self.train_accuracy_history = np.append( self.train_accuracy_history, actual_probs )
        else:
            self.test_accuracy_history = np.append( self.test_accuracy_history, actual_probs )

        return model_input_tensor
    
    def eval_and_train_model( self, train_data: np.ndarray ):
        """
        Evaluates and trains the model, adding the training data to the model history.
        
        :param train_data: Numpy array of training data
        :type train_data: np.ndarray
        """
        model_input_tensor = self.eval_model( train_data, is_train_data=True )
        self.train()

        train_len = train_data.shape[ 0 ]

        print( f"Training model..." )
        output_tensor = torch.tensor( train_data, dtype=torch.long ).to( device )

        dataset = TensorDataset( model_input_tensor, output_tensor )
        dataloader = DataLoader( dataset, batch_size=train_len )
        self.train_on_dataloader( dataloader, self.loss_fn, self.optimizer )
        print( f"Trained!" )

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
            print( f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]" )



class InteractableNeuralNetwork( NeuralNetwork ):
    def __init__( self, model_path: Path, epoch_len: int = 300, context_len: int = 5, model_width: int = 5, user_input_file: Path | None = None, lr: float = 1e-3, should_train: bool = False, eval_ratio: float = 0.1, weight_decay: float = 1e-4 ):
        super().__init__( context_len=context_len, model_width=model_width, lr=lr, weight_decay=weight_decay )
        self.rng = RandomNumberGenerator()
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
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Percentage (%) given by the model to the user\'s answer' )
        for label, accuracy_history in zip( [ 'train', 'test' ], [ self.train_accuracy_history, self.test_accuracy_history ] ):
            n_epochs = len( accuracy_history ) // self.epoch_len
            epochs = np.arange( 0, n_epochs, 1 )
            accuracy_epochs = accuracy_history[ :n_epochs * self.epoch_len ].reshape( -1, self.epoch_len )
            plt.errorbar( epochs, np.mean( accuracy_epochs, axis=1 ), np.std( accuracy_epochs, axis=1 ), label=label )
        plt.legend()
        plt.show()

    def r_fn( self ):
        self.training_loop( np.array( self.rng.next(), dtype=int ) )

    def repoch_fn( self ):
        self.training_loop( np.array( self.rng.next( self.epoch_len ), dtype=int ) )

    def read_fn( self ):
        if self.user_input_file is not None:
            with open( self.user_input_file ) as file:
                filedata = file.read()
            data = filedata.splitlines()

            # Ordering is important here to preserve context
            train_test_split_index = int( ( 1 - self.eval_ratio ) * len( data ) )
            train = data[ :train_test_split_index ]
            test = data[ train_test_split_index: ]
            self.eval_and_train_model( np.array( train, dtype=int ) )
            self.eval_model( np.array( test, dtype=int ) )
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
        if self.should_train:
            self.eval_and_train_model( input )
        else:
            self.eval_model( input )

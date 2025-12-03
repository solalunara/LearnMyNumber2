import os
import torch
from pathlib import Path
import numpy as np
from neuralnetwork import NeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


if __name__ == '__main__':
    print( "Initializing..." )

    # Try to load model from file, otherwise make it from scratch
    model_path = Path( 'model.pt' )
    user_input_file = Path( 'userinput.txt' )
    model = NeuralNetwork( model_path, user_input_file=user_input_file, lr=1e-1, epoch_len=10000 ).to( device )
    if model_path.exists():
        print( f"Loading pretrained model from {model_path}" )
        model.load_state_dict( torch.load( model_path, weights_only=True ) )

    # Main program loop
    print( "Please enter a number 0-9, type help for a list of commands, or q to close" )
    user_input = ""
    iter = 0
    auto_user_inputs = np.empty( (0) )
    while True:
        user_input = input( "> " )
        user_input = user_input.lower().strip()
        model.training_loop( user_input )



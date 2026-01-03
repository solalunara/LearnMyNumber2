import os
import torch
from pathlib import Path
import numpy as np
from neuralnetwork import NeuralNetwork, InteractableNeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


if __name__ == '__main__':
    print( "Initializing..." )

    # Try to load model from file, otherwise make it from scratch
    model_path = Path( 'model.pt' )
    user_input_file = Path( 'userinput.txt' )
    model = InteractableNeuralNetwork( model_path, user_input_file=user_input_file, lr=1e-2, epoch_len=10000, eval_ratio=0.5 ).to( device )
    if model_path.exists():
        print( f"Loading pretrained model from {model_path}" )
        model.load_state_dict( torch.load( model_path, weights_only=True, map_location=torch.device( device ) ) )

    # Main program loop
    print( "Please enter a number 0-9, type help for a list of commands, or q to close" )
    user_input = ""
    iter = 0
    auto_user_inputs = np.empty( (0) )
    while True:
        user_input = input( "> " )
        user_input = user_input.lower().strip()
        model.exec_command( user_input )



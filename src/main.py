import os
import torch
from pathlib import Path
import numpy as np
from neuralnetwork import NeuralNetwork, InteractableNeuralNetwork
import io
from tqdm import tqdm

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


if __name__ == '__main__':
    print( "Initializing..." )

    # Try to load model from file, otherwise make it from scratch
    model_path = Path( 'model2.pt' )
    user_input_file = Path( 'userinput.txt' )
    lr = 1e-3
    weight_decay = 1e-3
    model = InteractableNeuralNetwork( model_path, user_input_file=user_input_file, lr=lr, epoch_len=10000, eval_ratio=0.5, weight_decay=weight_decay ).to( device )
    if model_path.exists():
        print( f"Loading pretrained model from {model_path}" )
        model.load_state_dict( torch.load( model_path, weights_only=True, map_location=torch.device( device ) ) )

    # Save the initial weights so we can do two runs - one with random data and one with user data
    weights_buffer = io.BytesIO()
    torch.save( model.state_dict(), weights_buffer )


    # train the model
    for i in tqdm( range( 500 ), desc='read' ):
        model.exec_command( 'read' )

    # Reset the model, and the optimizer
    weights_buffer.seek( 0 )
    model.load_state_dict( torch.load( weights_buffer, weights_only=True, map_location=torch.device( device ) ) )
    model.optimizer = torch.optim.Adam( model.parameters(), lr=lr, weight_decay=weight_decay )
    model.optimizer.zero_grad()
    model.history = np.empty( (0), dtype=int )

    # RNG for reference
    for i in tqdm( range( 500 ), desc='repoch' ):
        model.exec_command( 'repoch' )


    # Main program loop
    print( "Please enter a number 0-9, type help for a list of commands, or q to close" )
    user_input = ""
    iter = 0
    auto_user_inputs = np.empty( (0) )
    while True:
        user_input = input( "> " )
        user_input = user_input.lower().strip()
        model.exec_command( user_input )



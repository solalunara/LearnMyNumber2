<h1>Learn My Number (2)</h1>

A modern re-implementation of my 2021 project "Learn My Number" using the pytorch library.

This model attempts to predict the next number a user will type given the context of the last 25 numbers.
The aim is to have a model that is able to simulate human pseudo-randomness. The user is prompted to enter random
numbers, and after filling up the context each number is used to train the model. The model can be saved and loaded
by the program.

A simple model which has been trained on a cyclical sequence from 0-9 repeating is included in this repository
as a proof of concept as cyclic.pt.

The command 'l' (load) currently reads the model from model.pt, as does the startup script for the program.
The command 'read' currently reads the file userinput.txt, assuming one user input per line. Other commands
can be listed by the 'help' command.

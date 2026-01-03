<h1>Learn My Number (2)</h1>

A modern re-implementation of my 2021 project "Learn My Number" using the pytorch library.

This model attempts to predict the next number a user will type given the context of the last 25 numbers.
The aim is to have a model that is able to simulate human pseudo-randomness. The user is prompted to enter random
numbers, and after filling up the context each number is used to train the model. The model can be saved and loaded
by the program.

The command 'l' (load) currently reads the model from model.pt, as does the startup script for the program.
The command 'read' currently reads the file userinput.txt, assuming one user input per line. Other commands
can be listed by the 'help' command.

The program splits the user input from the 'read' command - as currently set up in the repo the split is 50/50.
Included in the repository is a model trained on the full dataset repeated up to 160000 total data points,
with the training graph
![training graph](https://github.com/solalunara/LearnMyNumber2/blob/main/train_test_estim.png?raw=true)
where each epoch contains 10000 data points. With the current amount of training data the generalization process
appears to plateau around epoch 10. This training process included a weight decay of 1e-4 to reduce overfitting.

The model appears to be able to predict the evaluation dataset to a mean of better than random chance, but
with a high degree of uncertainty.


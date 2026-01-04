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
Included in the repository is a model trained on the full dataset up to 500 iterations.

![loss graph](https://github.com/solalunara/LearnMyNumber2/blob/main/loss.png?raw=true)

With the current amount of training data the generalization process appears to plateau around call 250 at a
learning rate of 1e-3, with the full dataset being trained on each iteration. This training process included
a weight decay of 1e-3 to attempt to reduce overfitting.

Included in the graph is also pseudo-random numbers from numpy passed to the model after the weights and biases
have been set to their pre-training values, as a control sample. This fails to improve above a perplexity
of 10, which is consistent with the setup of the problem as a classification between ten digits.

The model appears to be able to predict the evaluation dataset to a mean of better than random chance, but
prefers to overfit to the training data. The next steps are:

1 - Adding more human pseudo-random data (feel free to PR!)
2 - Switching architecture to an RNN
3 - Compare against other strategies (e.g. never repeating numbers, only pick most common number, etc)
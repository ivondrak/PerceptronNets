# This is a sample Python script.
from perceptron import Perceptron


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

training_set = [
    ([0.0, 1.0, 1.0], [0.0, 0.0]),
    ([1.0, 0.0, 0.0], [1.0, 1.0]),
    ([0.5, 0.5, 0.5], [1.0, 1.0])
]

net_input = [1.0, 0.0, 0.0]


def run_perceptron():

    # Use a breakpoint in the code line below to debug your script.
    perceptron = Perceptron(training_set, 1000)
    perceptron.learning()
    output = perceptron.run([net_input])
    print("Result is: ", output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_perceptron()


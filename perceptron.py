import numpy as np

class Perceptron:
    def __init__(self, training_set, epochs):
        self.training_data = training_set
        self.num_features = len(training_set[0][0])
        self.num_results = len(training_set[0][1])
        self.weights = np.random.randn(self.num_results, self.num_features + 1)
        self.learning_rate = 0.3
        self.epochs = epochs

    def signum_function(self, x):
        return (x >= 0).astype(np.int32)

    def learning(self):
        for epoch in range(self.epochs):
            for vector in self.training_data:
                input_vector = np.array(vector[0])
                desired_output = np.array(vector[1])
                actual_input = np.insert(input_vector, 0, -1.0)
                product = np.dot(self.weights, actual_input)
                actual_output = self.signum_function(product)
                error = desired_output - actual_output
                self.weights += np.dot(self.learning_rate * error.reshape([self.num_results, 1]), actual_input.reshape([1, self.num_features + 1]))

    def run(self, net_input):
        actual_input = np.insert(net_input, 0, -1.0)
        product = np.dot(self.weights, actual_input)
        return self.signum_function(product).tolist()
    
    def calculate_mse(self):
        mse = 0
        for vector in self.training_data:
            input_vector = np.array(vector[0])
            desired_output = np.array(vector[1])
            actual_input = np.insert(input_vector, 0, -1.0)
            product = np.dot(self.weights, actual_input)
            actual_output = self.signum_function(product)
            error = desired_output - actual_output
            mse += np.sum(error ** 2)
        mse /= len(self.training_data)
        return mse
    
    def calculate_max(self):
        max_error = 0
        for vector in self.training_data:
            input_vector = np.array(vector[0])
            desired_output = np.array(vector[1])
            actual_input = np.insert(input_vector, 0, -1.0)
            product = np.dot(self.weights, actual_input)
            actual_output = self.signum_function(product)
            error = desired_output - actual_output
            max_error = max(max_error, np.max(np.abs(error)))
        return max_error


    
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot(training_errors, validation_errors):
    iterations = [i + 1 for i in range(len(training_errors))]

    # Plot training error
    plt.plot(iterations, training_errors, label='Training error')

    # Plot  validation error
    plt.plot(
        [i + 1 for i in range(0, len(validation_errors)*100, 100)],
        validation_errors, label='Validation error'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Weight Decay: Training and Testing errors')

    plt.legend()

    plt.show()



class MLP:

    def __init__(self, no_hidden_nodes, learning_parameter): 
        self.hidden_node_weights = np.random.uniform(-2/5, 2/5, size=(no_hidden_nodes, 5))
        self.hidden_node_biases = np.random.uniform(-2/5, 2/5, size=(no_hidden_nodes, 1))
        self.output_node_weights = np.random.uniform(-2/5, 2/5, size=no_hidden_nodes)
        self.output_node_bias = np.random.uniform(-2/5, 2/5)
        self.no_hidden_nodes = no_hidden_nodes
        self.learning_parameter = learning_parameter

    # compute weighted sums for every node
    def forward_pass(self, input_nodes):
        # get weighted sums by calculating dot product of input nodes and their weights
        #  then adding the hidden node biases
        hidden_node_weighted_sums = (
            np.dot(self.hidden_node_weights, input_nodes) + self.hidden_node_biases
        )
        '''
        print("input nodes . input weights")
        print(np.dot(self.hidden_node_weights, input_nodes))
        print("hidden node biases")
        print(self.hidden_node_biases)'''
        #apply sigmoid function
        hidden_node_outputs = sigmoid(hidden_node_weighted_sums) 
        '''print("Weighted Sums:\n%s\nActivations:\n%s" % (hidden_node_weighted_sums, hidden_node_outputs ))'''

        # dot product of output weights and activations
        output_node_weighted_sum = np.dot(self.output_node_weights,hidden_node_outputs) + self.output_node_bias
        #return output activation
        return hidden_node_outputs, sigmoid(output_node_weighted_sum)

    # pass in input nodes aswell as pane target value
    def backwards_pass(self, epoch, inputs_and_target, hidden_node_outputs, predicted_PanE_value):
        # calculate delta of the output node by getting the difference between the correct and predicted output
        output_node_sigmoid_derivative = predicted_PanE_value * (1 - predicted_PanE_value)

        # ! get regularisation parameter
        regularisation_parameter = 1 / (epoch * self.learning_parameter)

        if regularisation_parameter < 0.001:
            regularisation_parameter = 0.001
        elif 0.1 < regularisation_parameter:
            regularisation_parameter = 0.1

        # ! get omega
        omega = (
            sum(w**2 for node in self.hidden_node_weights for w in node)
            + sum(w**2 for w in self.output_node_weights)
            + sum(w**2 for w in self.hidden_node_biases)
            + self.output_node_bias**2
        ) / (2 * (6*self.no_hidden_nodes + self.no_hidden_nodes + 1))

        # ! get error
        error = (inputs_and_target[-1][0] - predicted_PanE_value) + regularisation_parameter * omega
        
        output_node_delta = (inputs_and_target[-1][0] - predicted_PanE_value)*output_node_sigmoid_derivative
        # calculate delta of each hidden node by multiplying...
        # ...[the weight of its connection to the output node] with [the delta of the output node]...
        # ...and the derivative equation for the hidden layer
        hidden_node_deltas =[]
        for i in range(self.no_hidden_nodes):
            hidden_node_sigmoid_derivative = hidden_node_outputs[i] * (1 - hidden_node_outputs[i])
            hidden_node_deltas.append(self.output_node_weights[i] * output_node_delta * hidden_node_sigmoid_derivative)
                        
        # ammend all weights by adding the product of (learning parameter, delta of node, output of the node) to the weight
        for i in range(self.no_hidden_nodes):
            # update hidden node weights and biases
            for j in range(5):
                self.hidden_node_weights[i][j] += self.learning_parameter * hidden_node_deltas[i] * inputs_and_target[j][0]
                
            self.hidden_node_biases[i][0] += self.learning_parameter * hidden_node_deltas[i]
                
            # next ammend output weights and output biases
            self.output_node_weights[i] += self.learning_parameter * output_node_delta * hidden_node_outputs[i]
            
        self.output_node_bias += self.learning_parameter * output_node_delta
        
    def train(self, no_epochs, input_nodes):
        training_errors = []
        validation_errors = []

        for x in range(no_epochs):
            sum_of_squared_differences = 0
            for entry in input_nodes:
                # pass in a row of entry data from excel file excluding pane data
                hidden_node_outputs, output_node_activation = self.forward_pass(entry[: -1])
                # get sum of squared differences to use to calculate the rmse (the error value at the current epoch)
                sum_of_squared_differences += (output_node_activation - entry[-1])**2
                # call backwards pass and update weights
                self.backwards_pass(1 + x, entry, hidden_node_outputs, output_node_activation)
            # calculate the error value at the current epoch
            error = math.sqrt(sum_of_squared_differences / len(input_nodes))
            training_errors.append(error)

            # at regular intervals run forward_pass to validate the model
            if not x % 100 or x == no_epochs - 1:
                print(x)
                sum_of_squared_differences = 0
                for entry in validation_input_nodes:
                    # pass in a row of entry data from excel file excluding pane data
                    hidden_node_outputs, output_node_activation = self.forward_pass(entry[: -1])
                    # get sum of squared differences to use to calculate the rmse (the error value at the current epoch)
                    sum_of_squared_differences += (output_node_activation - entry[-1])**2
                # calculate the error value at the current epoch
                error = math.sqrt(sum_of_squared_differences / len(validation_input_nodes))
                validation_errors.append(error)

        plot(training_errors, validation_errors)
        #print(training_errors[-1], validation_errors[-1])


if __name__ == "__main__":  
    mlp = MLP(3, 0.01)
    data = pd.read_excel('DataSet - Training.xlsx')

    validation_data = pd.read_excel('DataSet - Testing.xlsx')
    validation_input_nodes = []
    for index, row in validation_data.iterrows():
        validation_input_nodes.append([[row['T']], [row['W']], [row['SR']], [row['DSP']], [row['DRH']], [row['PanE']]])    # Convert the data into the required format
    input_nodes = []
    for index, row in data.iterrows():
        input_nodes.append([[row['T']], [row['W']], [row['SR']], [row['DSP']], [row['DRH']], [row['PanE']]])
    mlp.train(1000, input_nodes)

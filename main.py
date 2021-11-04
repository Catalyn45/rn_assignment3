import gzip
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List
import random
from scipy.special import softmax


def activation(weights: np.array, biases: np.array, data: np.array):
    multiplied = weights * data
    return np.sum(multiplied, axis=1) + biases


def softmax(activations):
    e_x = np.exp(activations - np.max(activations))
    return e_x / e_x.sum(axis=0)


def softmax_derivative(activations):
    s = activations.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


def sigmoid(activations):
	return 1.0 / (np.exp(-1 * activations) + 1.0)


def sigmoid_derivative(activations):
	return activations * (1.0 - activations)


class Layer:
    def __init__(self, first_nodes, second_nodes) -> None:
        self.weights: np.array  = np.random.normal(0, 0.1, (second_nodes, first_nodes))
        self.biases: np.array = np.random.normal(0, 0.1, second_nodes)
        self.outputs: np.array = np.zeros(second_nodes)
        self.deltas: np.array = np.zeros(second_nodes)


class Network:
    def __init__(self, *args) -> None:
        self.layers:List[Layer] = [Layer(first_layer, second_layer) for first_layer, second_layer in zip(args[:-1], args[1:])]

    def forward_propagation(self, data: np.array):    
        for i, layer in enumerate(self.layers):
            activations = activation(layer.weights, layer.biases, data if i == 0 else self.layers[i-1].outputs)
            layer.outputs = sigmoid(activations) if i != len(self.layers) - 1 else softmax(activations)

    def train(self, train_set, valid_set, learning_rate):
        images, labels = train_set
        samples = list(zip(images, labels))

        epoch = 0

        while True:
            random.shuffle(samples)
            for image_index, (image, label) in enumerate(samples):
                self.forward_propagation(image)

                expected = np.zeros(10)
                expected[label] = 1

                for i, layer in reversed(list(enumerate(self.layers))):
                    if i == len(self.layers) - 1:
                        errors = expected - layer.outputs
                        transfer_derivative_function = softmax
                    else:
                        transfer_derivative_function = sigmoid_derivative
                        errors = self.layers[i + 1].weights.transpose().dot(self.layers[i + 1].deltas[:, np.newaxis]).flatten()
                    
                    layer.deltas = errors * transfer_derivative_function(layer.outputs)

                for i, layer in enumerate(self.layers):
                    previous_activation = image if i == 0 else self.layers[i - 1].outputs
                    layer.weights += np.matrix(previous_activation).transpose().dot(np.matrix(layer.deltas) * learning_rate).transpose()
                    layer.biases += layer.deltas * learning_rate

            rate = self.test(valid_set)
            print(f'Epoch: {epoch} accuracy: {round(rate * 100, 2)} %')

            learning_rate *= .98
            epoch += 1

    def test(self, test_set):
        success = 0
        failed = 0

        test_images, test_labels = test_set

        for image_index, (image, label) in enumerate(zip(test_images, test_labels)):
            self.forward_propagation(image)

            expected = np.zeros(10)
            expected[label] = 1

            max_index = np.argmax(self.layers[-1].outputs)

            if max_index == label:
                success += 1
            else:
                failed += 1

        rate = success / (success + failed)

        return rate


def get_datasets(file_name):
    with gzip.open(file_name, "rb") as file:
        return pickle.load(file, encoding='latin')


def main():
    train_set, valid_set, test_set = get_datasets("mnist.pkl.gz")
    network = Network(784, 100, 10)
    network.train(train_set, valid_set, learning_rate=0.3)


if __name__ == '__main__':
    main()

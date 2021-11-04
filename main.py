import gzip
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List
import random


class Layer:
    def __init__(self, first_nodes, second_nodes) -> None:
        self.weights: np.array  = np.random.sample((second_nodes, first_nodes))
        self.biases: np.array = np.random.sample(second_nodes)
        self.outputs: np.array = np.zeros(second_nodes)
        self.deltas: np.array = np.zeros(second_nodes)


class Network:
    def __init__(self, *args) -> None:
        self.layers:List[Layer] = [Layer(first_layer, second_layer) for first_layer, second_layer in zip(args[:-1], args[1:])]


def activation(weights: np.array, biases: np.array, data: np.array):
    return np.sum(weights * data, axis=1) + biases


# sigmoid
def transfer(activations):
	return 1.0 / (1.0 + np.exp(-activations))


# sigmoid derivative
def transfer_derivative(outputs):
	return outputs * (1.0 - outputs)


def forward_propagation(network: Network, data: np.array):    
    for i, layer in enumerate(network.layers):
        layer.outputs = transfer(activation(layer.weights, layer.biases, data if i == 0 else network.layers[i-1].outputs))


def get_datasets(file_name):
    with gzip.open(file_name, "rb") as file:
        return pickle.load(file, encoding='latin')


def train(network: Network, train_set, valid_set, learning_rate):
    images, labels = train_set
    samples = list(zip(images, labels))

    epoch = 0

    while True:
        random.shuffle(samples)
        for image_index, (image, label) in enumerate(samples):
            forward_propagation(network, image)

            expected = np.zeros(10)
            expected[label] = 1

            for i, layer in reversed(list(enumerate(network.layers))):
                if i == len(network.layers) - 1:
                    errors = expected - layer.outputs
                else:
                    errors = network.layers[i + 1].weights * (network.layers[i + 1].deltas[:, np.newaxis]), axis=1)
                
                layer.deltas = np.array(errors) * transfer_derivative(layer.outputs)

            for i, layer in enumerate(network.layers):
                previous_activation = image if i == 0 else network.layers[i + 1]
                layer.weights += np.matrix(previous_activation).transpose().dot(layer.deltas) * learning_rate
                layer.biases += layer.deltas * learning_rate

        rate = test(network, valid_set)
        print(f'Epoch: {epoch} accuracy: {round(rate * 100, 2)} %')

        learning_rate *= .98
        epoch += 1


def test(network, test_set):
    success = 0
    failed = 0

    test_images, test_labels = test_set

    for image_index, (image, label) in enumerate(zip(test_images, test_labels)):
        forward_propagation(network, image)

        expected = np.zeros(10)
        expected[label] = 1

        max_index = np.where(network.layers[-1].outputs == network.layers[-1].outputs.max())[0]

        if max_index == label:
            success += 1
        else:
            failed += 1

    rate = success / (success + failed)

    return rate


def main():
    train_set, valid_set, test_set = get_datasets("mnist.pkl.gz")
    network = Network(784, 100, 10)
    train(network, train_set, valid_set, learning_rate=0.001)


if __name__ == '__main__':
    main()

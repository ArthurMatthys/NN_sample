#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 arthur <arthur@arthur-Lenovo-Y520-15IKBM>
#
# Distributed under terms of the MIT license.


import sys
import argparse
import numpy as np
from tqdm import tqdm


class NeuralNetwork():
    def __init__(self, matrix_input, output, array_layer, loading_file, regularisation, alpha):
        self.nbr_layer = len(array_layer) - 1
        self.nbr_input = matrix_input.shape[0]
        self.matrix_ones = np.float32(np.ones((self.nbr_input,1)))
        self.output = output
        self.size_layer = array_layer
        self.neural_matrix = [np.hstack((self.matrix_ones, np.float32(matrix_input)))]
        self.__create_weight_matrix__(array_layer, loading_file)
        self.alpha = np.float32(alpha)
        self.dynamic_alpha = 1 if alpha == 0 else 0
        self.regularisation = regularisation
        self.epsilon = 10 ** -5
        self.gradient = self.weight_matrix.copy()


    def __create_weight_matrix__(self, array_layer, loading_file):
        if loading_file == "":
            self.weight_matrix = []
            for i in range(1, len(array_layer)):
                previous = array_layer[i - 1]
                actual = array_layer[i]
                min_max = np.sqrt(6 / (previous + actual))
                self.weight_matrix.append(np.float32(np.random.uniform(-min_max, min_max, (previous + 1, actual))))
        else:
            self.weight_matrix = np.load(loading_file)


    def forward_propagation(self):
        for i, matrix_weight in enumerate(self.weight_matrix):
            new_matrix = 0.99998 / (1 + np.exp(-np.clip(np.dot(self.neural_matrix[i], matrix_weight),-50, 50))) + 0.00001
            if (i != self.nbr_layer - 1):
                new_matrix = np.hstack((self.matrix_ones, new_matrix))
            self.neural_matrix.append(new_matrix)
                

    def cost_function(self):
        actual_output = self.neural_matrix[-1].copy()
        actual_output = np.clip(actual_output, self.epsilon, 1 - self.epsilon, actual_output)
        self.cost = -np.sum(np.multiply(self.output, np.log(actual_output)) + np.multiply((1 - self.output), np.log(1 - actual_output)))
        self.cost /= self.nbr_input



    def regularized_cost_function(self):
        self.cost_function()
        sum_matrix = 0
        for matrix_weight in self.weight_matrix:
            sum_matrix += np.sum(np.square(matrix_weight[1:]))
        self.regularized_cost = self.cost + (self.regularisation / (self.nbr_layer * self.nbr_input)) * sum_matrix
        if self.dynamic_alpha == 1:
            self.alpha = np.float32(3 / (self.regularized_cost + 0.1))



    def accuracy_function(self):
        output_matrix = self.neural_matrix[-1]
        self.accuracy = np.mean(np.equal(np.argmax(self.output, axis = 1), np.argmax(output_matrix, axis = 1)))


    def backward_propagation(self):
        gradient = []
        middle_part = [self.neural_matrix[-1] - self.output]
        for i in range(self.nbr_layer, 1,-1):
            actual = self.neural_matrix[i - 1][:,1:]
            middle_part.append(np.dot(middle_part[-1], np.transpose(self.weight_matrix[i - 1][1:])) * actual * (1 - actual))
        for i in range(self.nbr_layer):
            self.gradient[i] = np.dot(np.transpose(self.neural_matrix[i]), middle_part[-i - 1]) * self.alpha / self.nbr_input
            self.gradient[i][1:] += self.regularisation * self.weight_matrix[i][1:] / self.nbr_input
        


    def apply_gradient(self):
        for i in range(len(self.gradient)):
            self.weight_matrix[i] -= self.gradient[i]



    def run(self, nbr_iteration, saving_file):
        for i in tqdm(range(nbr_iteration)):
            if i == 0:
                tqdm.write(" ____________________________________________________________")
                tqdm.write("|  iteration  |     cost     |   accuracy   |     alpha      |")
                tqdm.write(" ____________________________________________________________")
            self.forward_propagation()
            self.backward_propagation() 
            self.regularized_cost_function()
            self.apply_gradient()
            if (i % 50 == 0 or i == nbr_iteration - 1):
                self.accuracy_function()
                tqdm.write("|   {:5}     |  {:.8f}  |  {:.8f}  |   {:.7f}    |".format(i + 1 if i == nbr_iteration - 1 else i, self.regularized_cost, self.accuracy, self.alpha))
            self.neural_matrix = [self.neural_matrix[0]]
        if saving_file != "":
            np.save(saving_file, self.weight_matrix)




parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="The name of the input file. It need to be stored so that np.load can properly load the input")
parser.add_argument("output", type=str, help="The name of the expected output file. It need to be stored so that np.load can properly load the expected output")
parser.add_argument("size_layer", nargs='+', type=int, help="The size of each layer")
parser.add_argument("--save", type=str, help="Name of the saving file for weigths", default="")
parser.add_argument("--load", type=str, help="Name of the loading file for weigths", default="")
parser.add_argument("--iteration", "-i", type=int, help="Number of iterations that need to be done", default=1000)
parser.add_argument("--regularisation", "-r", type=float, help="The value of the regularisation in the cost", default=0.016)
parser.add_argument("--alpha", type=float, help="the value of alpha. By default, alpha is dynamic", default = 0)

args = parser.parse_args()

input_matrix = np.load(args.input)
output = np.load(args.output)
output_matrix = np.float32(np.zeros((output.shape[0], int(np.max(output)) + 1)))
for i in range(len(output)):
    output_matrix[i][int(output[i])] = 1

network = NeuralNetwork(input_matrix, output_matrix, args.size_layer, args.load, np.float32(args.regularisation), args.alpha)


network.run(args.iteration, args.save)





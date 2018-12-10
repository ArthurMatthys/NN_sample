#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 arthur <arthur@arthur-Lenovo-Y520-15IKBM>
#
# Distributed under terms of the MIT license.

import numpy as np
import sys

#the matrix of input is the matrix where each columns correspond to one input
#the array_size_layer correspond to the size of each invisible layer


class NeuralNetwork:
    def __init__(self, matrix_input, array_size_layer, expected_output, round_trip, adjust_regularisation = 1, save = 1, load_matrix = 0):
        self.save = save
        self.nbr_input = len(matrix_input)
        self.input = np.float32(np.hstack((matrix_input, np.ones((self.nbr_input, 1)))))
        self.neural_matrix = [self.input]
        self.expected_output = np.float32(expected_output)
        self.to_compare_output = []
        self.__create_compare_output()
        self.__create_ones()
        self.array_matrix = []
        if (load_matrix):
            self.array_matrix = np.load("thetas1.npy")
        else:
            self.__create_array_matrix(array_size_layer)
        self.nbr_layers = len(self.array_matrix)
        self.adjust_regularisation = adjust_regularisation / self.nbr_input
        self.round_trip = round_trip
        self.gradient = []

    def __create_array_matrix(self, array):
        tmp = len(self.neural_matrix[0][0]) - 1
        temp = [425,35]
        for i in range(len(array)):
            max_min = np.sqrt(6 / temp[i]) 
            matrix = np.float32(np.random.uniform(-max_min, max_min, (tmp + 1, array[i]))) 
            self.array_matrix.append(matrix)
            tmp = array[i]
        max_min = np.sqrt(6 / temp[1]) 
        self.array_matrix.append(np.float32((np.random.uniform(-max_min, max_min, (tmp + 1, len(self.expected_output[0]))))))


    def __create_compare_output(self):
        self.to_compare_output = np.float32(np.zeros((self.nbr_input)))
        for i in range(self.nbr_input):
            self.to_compare_output[i] = i // 500
    

    def __create_ones(self):
        tmp = len(self.neural_matrix[0][0]) - 1
        self.matrix_ones = np.float32(np.ones((self.nbr_input,1)))


    def forward_propagation(self):
        for i in range(self.nbr_layers):
            new_matrix = 1 / (1 + np.exp(-np.dot(self.neural_matrix[i], self.array_matrix[i])))
            self.neural_matrix.append(np.hstack((new_matrix, self.matrix_ones)))


    def compare_output(self):
        epsilon = 1*(10**-5)
        output = self.last_matrix.copy()
        np.clip(output, epsilon, 1 - epsilon, output)
        self.cost = -np.sum(np.multiply(self.expected_output, np.log(output)) + np.multiply((1 - self.expected_output), np.log(1 - output)))
        self.cost /= self.nbr_input


    def regularized_cost_function(self):
        tot = 0
        self.compare_output()
        for i in self.array_matrix:
            tot += np.sum(np.square(i)[:,:-1])
        self.regularized_cost = self.cost + (self.adjust_regularisation / 2) * tot 


    def find_accuracy(self):
        array_max = np.argmax(self.last_matrix, axis = 1)
        tot = np.equal(array_max, self.to_compare_output)
        self.accuracy = np.mean(tot)


    def backward_propagation(self, regularisation):
        self.last_matrix = self.neural_matrix[-1][:,:-1]
        middle_part = self.last_matrix - self.expected_output
        for i in range(self.nbr_layers)[::-1]:
            matrix_weight = self.array_matrix[i].copy()
            matrix_weight[:,-1] = 0
            grad = np.dot(np.transpose(self.neural_matrix[i]), np.multiply(middle_part, (1 / self.nbr_input)))
            grad += np.multiply(regularisation, matrix_weight)
            self.gradient.append(grad)
            actual_layer = self.neural_matrix[i][:,:-1].copy()
            actual_layer-= np.square(actual_layer)
            cropped_transpose = np.transpose(self.array_matrix[i][:-1])
            middle_part = np.multiply(actual_layer, np.dot(middle_part, cropped_transpose))

 
    def apply_gradient(self):
        for i in range(self.nbr_layers):
            grad = self.gradient[self.nbr_layers - i - 1]
            alpha = 1
            self.array_matrix[i] -= (np.multiply(alpha, grad))


    def run_network(self):
        for i in range(self.round_trip + 1):
            self.forward_propagation()
            self.backward_propagation(self.adjust_regularisation)
            self.apply_gradient()
            if (i % 50 == 0):
                self.regularized_cost_function()
                self.find_accuracy()
                print(i, self.regularized_cost, self.accuracy)
            self.neural_matrix = [self.input]
            self.gradient = []
        if self.save:
            np.save("layers", self.array_matrix)

    
    def test(self, input_test, expected_output_test):
        output = np.hstack((input_test, [[1]] * len(input_test)))
        for i in self.array_matrix:
            output = self.normalize(np.dot(output, i))
            output = np.hstack((output, [[1]] * len(output))) 
        output = np.argmax(output[:,:-1], axis = 1)
        tot = 0
        for i in range(len(expected_output_test)):
            if (expected_output_test[i] == output[i]):
                tot += 1
        return (tot / len(input_test))

a = np.load("dataset.npy")

output = np.zeros([5000, 10])
for i in range(len(output)):
    output[i][i//500] = 1

#stat = 1
#b = 0
#for b in range(501):
#    test = []
#    entry = []
#    expected_output_test = []

#    for i in range(len(a)):
#        if b <= i % 500 < b + 25:
#            test.append(a[i])
#        else:
#            entry.append(a[i])


#    for i in range(len(test)):
#        expected_output_test.append(i // 25)



network = NeuralNetwork(a, [25], output, int(sys.argv[1]) if len(sys.argv) > 2 else 5, float(sys.argv[2]) if len(sys.argv) > 2 else 0.1, 0, 0)


network.run_network()
#valid_test = network.test(test, expected_output_test)
#print(valid_test)
#stat = 1



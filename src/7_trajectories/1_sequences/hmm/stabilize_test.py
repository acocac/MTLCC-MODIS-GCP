#!/bin/env python

"""
A simple unit test for the Markov chain stabilization code.

Author: Parker Abercrombie <parker@pabercrombie.com>
"""

import sys
import numpy as np

import hmm

probs = np.array([0.0, # First element is land/water mask
                  0.0, 0.815, 0.0, 0.144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041, 0.0, 0.0, 0.0,
                  0.0, 1.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.818, 0.0, 0.038, 0.0, 0.0, 0.0, 0.043, 0.0, 0.0, 0.045, 0.0, 0.0, 0.057, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.214, 0.0, 0.0, 0.0, 0.341, 0.263, 0.0, 0.060, 0.121, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.123, 0.0, 0.0, 0.0, 0.0, 0.301, 0.186, 0.0, 0.251, 0.0, 0.138, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.200, 0.0, 0.0, 0.0, 0.0, 0.0, 0.228, 0.0, 0.572, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.399, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.600, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.218, 0.0, 0.0, 0.0, 0.541, 0.0, 0.240, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.413, 0.0, 0.0, 0.587, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.195, 0.0, 0.0, 0.523, 0.0, 0.0, 0.281, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.739, 0.0, 0.260, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0])

expected = np.array([2, 2, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12])

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=1000)

    n_labels = 17
    change_prob = 0.1
    transition_matrix = np.empty((n_labels, n_labels))
    transition_matrix.fill(change_prob)
    np.fill_diagonal(transition_matrix, 1.0 - change_prob)

    labels = hmm.viterbi(probs, n_labels, transition_matrix)
    if np.array_equal(labels, expected):
        print "TEST PASSED"
    else:
        print "TEST FAILED"
        print "Expected: ", expected
        print "Actual: ", labels

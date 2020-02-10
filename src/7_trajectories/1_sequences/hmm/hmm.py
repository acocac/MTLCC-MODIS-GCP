import numpy as np

"""
Utilities for working with Hidden Markov Models (HMMs).

Author: Parker Abercrombie <parker@pabercrombie.com>
"""

NO_LABEL = 255

# Pixels identified as water in the land/water mask are not processed
# with the Markov chain. For these pixels, just return a precomputed
# vector.
WATER_PROB_VECTOR = np.zeros((12, 17))
WATER_PROB_VECTOR[:, 16] = 1.0
WATER_PROB_VECTOR.reshape(12 * 17)

def normalize(v):
    """
    Normalize a probability vector by dividing each element by the
    sum of the elements. The elements are probabilities, which are
    assumed to be in the range [0, 1]. Returns v unmodified if the sum
    is <= 0.0.
    """
    Z = v.sum()
    if Z > 0.0:
        return v / Z
    return v

def forward(time_series, n_labels, transition_matrix):
    """
    Use the Forward algorithm to compute marginal probabilities by
    propgating influence forward along the chain. For background on this
    algorithm see Section 17.4.2 of 'Machine Learning: A Probablistic
    Perspective' by Kevin Murphy.
    """

    lw_mask = time_series[0]
    if lw_mask == 1:
        WATER_PROB_VECTOR

    # Reshape data to a NxK matrix where N is number of time steps and
    # K is number of labels. Each row represents one time step.
    time_series = time_series[1:].reshape((-1, n_labels))

    n_steps = time_series.shape[0]

    belief = np.empty((n_steps, n_labels))

    belief[0, :] = time_series[0, :]

    for t in range(1, n_steps):
        v = np.multiply(time_series[t, :], transition_matrix.transpose().dot(belief[t - 1, :]))
        belief[t, :] = normalize(v)

    # Return belief as flattened vector
    return belief.reshape(n_steps * n_labels)

def forward_mle(time_series, n_labels, transition_matrix):
    """
    Run the forward algorithm to compute marginal probabilities for
    each year in the time series, and then return the maximum
    likelihood label for each year.
    """
    probs = forward_backward(time_series, n_labels, transition_matrix)
    probs = probs.reshape((-1, n_labels))
    labels = np.argmax(probs, axis=1) + 1
    return labels

def forward_backward(time_series, n_labels, transition_matrix):
    """
    Use the Forward/Backward algorithm to compute marginal probabilities by
    propgating influence forward along the chain. For background on this
    algorithm see Section 17.4.2 of 'Machine Learning: A Probablistic
    Perspective' by Kevin Murphy.
    """

    lw_mask = time_series[0]
    if lw_mask == 1:
        WATER_PROB_VECTOR

    # Reshape data to a NxK matrix where N is number of time steps and
    # K is number of labels. Each row represents one time step.
    time_series = time_series[1:].reshape((-1, n_labels))

    n_steps = time_series.shape[0]

    forward = np.empty((n_steps, n_labels))
    backward = np.empty((n_steps, n_labels))

    # Compute forward messages
    forward[0, :] = time_series[0, :]
    for t in range(1, n_steps):
        v = np.multiply(time_series[t, :], transition_matrix.transpose().dot(forward[t - 1, :]))
        forward[t, :] = normalize(v)

    # Compute backward messages
    backward[n_steps - 1, :] = np.ones(n_labels)

    for t in range(n_steps - 1, 0, -1):
        v = np.dot(transition_matrix, np.multiply(time_series[t, :], backward[t, :]))
        backward[t - 1, :] = normalize(v)

    belief = np.multiply(forward, backward)
    Z = np.sum(belief, axis=1)
    Z[Z == 0.0] = 1.0       # Ignore zero entries
    belief = np.divide(belief, Z.reshape((n_steps, 1))) # Normalize

    # Return belief as flattened vector
    return belief.reshape(n_steps * n_labels)

def forward_backward_mle(time_series, n_labels, transition_matrix):
    """
    Run the forward/backward algorithm to compute marginal
    probabilities for each year in the time series, and then return
    the maximum likelihood label for each year.
    """
    probs = forward_backward(time_series, n_labels, transition_matrix)
    probs = probs.reshape((-1, n_labels))

    labels = np.argmax(probs, axis=1) + 1
    return labels

def viterbi(time_series, n_labels, transition_matrix, vectorized=True):
    """
    Use the Viterbi algorithm to determine the most likely series
    of states from a time series. For background on this algorithm see
    http://en.wikipedia.org/wiki/Viterbi_algorithm, and Section 17.4.4 of
    'Machine Learning: A Probablistic Perspective' by Kevin Murphy.
    """

    lw_mask = time_series[0]
    if lw_mask == 1:
        WATER_PROB_VECTOR

    # Reshape data to a NxK matrix where N is number of time steps and
    # K is number of labels. Each row represents one time step.
    time_series = time_series[1:].reshape((-1, n_labels))

    n_steps = time_series.shape[0]

    # Work in log domain to avoid numerical underflow. Add a small
    # positive value to avoid problems taking log of zero.
    ln_time_series = np.log(time_series + np.finfo(np.float).eps)
    ln_transition_matrix = np.log(transition_matrix + np.finfo(np.float).eps)

    belief = np.empty((n_steps, n_labels))
    backtrack = np.empty((n_steps, n_labels), dtype=np.int)

    belief[0, :] = ln_time_series[0, :]

    if vectorized:
        for t in range(1, n_steps):
            local_belief = belief[t - 1, :] + ln_transition_matrix + ln_time_series[t, :].reshape((n_labels, 1))
            local_belief.max(axis=1, out=belief[t, :])
            local_belief.argmax(axis=1, out=backtrack[t, :])
    else:
        # A naive implementation of the algorithm. This
        # implementation is much slower, but is included for
        # testing and debugging, as it is easier to understand
        # than the vectorized version above.
        for t in range(1, n_steps):
            for k in range(n_labels):
                local_belief = np.multiply(belief[t - 1, :], transition_matrix[:, k]) * time_series[t, k]
                belief[t, k] = np.max(local_belief)
                backtrack[t, k] = np.argmax(local_belief)

    # Backtrack to find most likely state sequence
    path = np.empty((n_steps), dtype=np.uint8)
    max_belief = np.empty((n_steps), dtype=np.uint8)

    path[n_steps - 1] = np.argmax(belief[n_steps - 1, :])
    max_belief[n_steps - 1] = np.max(belief[n_steps - 1, :])

    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backtrack[t, path[t]]

    path += 1 # Add one so that the labels begin at one

    # Set any step where the belief if less than a threshold to
    # NO_LABEL
    for t in range(n_steps):
        if belief[t, path[t] - 1] == 0.0: # TODO floating point comparison to zero is sketchy. Should be based on tolerance
            path[t] = NO_LABEL

    return path

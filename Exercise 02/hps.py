import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import heapq

# challenge question
# method 01: natural solution by sorting the list
# signature: posteriors, threshold -> index set
# overall complexity:
# time: O(nlogn + 2n)
# space: O(n)
def highest_prob_set(posterior_probabilities, alpha):
    n = len(posterior_probabilities)
    # create hashmap that maps prob -> index
    # time complexity O(n) and space complexity O(n)
    dict = {}
    for i in range(n):
        # set k-v pair as (prob, index)
        p = posterior_probabilities[i]
        dict[p] = i
    # sort complexity: O(nlogn)
    posterior_probabilities = sort(posterior_probabilities, decreasing = TRUE)

    # linear search complexity: O(n)
    j = 1
    sum = 0
    hps = []
    while j <= n and sum < alpha:
        p = posterior_probabilities[j]
        hps.append(dict[p])
        j += 1
        sum += p

    return (hps)






def highest_prob_set(posterior_probabilities, alpha):
    # Max-heap requires negating numbers because Python's heapq is a min-heap
    max_heap = [(-value, index) for index, value in enumerate(posterior_probabilities)]
    heapq.heapify(max_heap)  # Transform the list into a heap
    print("max_heap is:", max_heap)

    running_sum = 0
    subset_indices = []

    while running_sum < alpha and max_heap:
        value, index = heapq.heappop(max_heap)  # Extract max element (negated)
        running_sum += -value  # Add the original value
        subset_indices.append(index)

    return subset_indices


# method 02: using priority queue (max heap)
# signature: posteriors, threshold -> index set
# overall complexity:
# time: at worst O(nlogn + n)
# space: O(n)

# test
posterior_probabilities = [
0.000000e+00, 8.962180e-06, 1.358478e-04, 6.495222e-04, 1.932057e-03,
4.422128e-03, 8.558410e-03, 1.472297e-02, 2.318469e-02, 3.404260e-02, 
4.716937e-02, 6.215460e-02, 7.824832e-02, 9.430430e-02, 1.087235e-01, 
1.193975e-01, 1.236517e-01, 1.181890e-01, 9.903303e-02, 6.147159e-02,
0.000000e+00]

print(highest_prob_set(posterior_probabilities, alpha=0.75))


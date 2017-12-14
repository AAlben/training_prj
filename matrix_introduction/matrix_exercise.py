import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


values = range(1, 11)

n_array = np.array((1, 10))
for index in values:
    n_array[0][index - 1] = index

distance_result = pairwise_distances(n_array, metric='cosine')

print(distance_result)
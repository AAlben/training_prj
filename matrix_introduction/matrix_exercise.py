import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


values = range(1, 11)

n_array = np.array(values)

distance_result = pairwise_distances(n_array, metric='cosine')

print(distance_result)
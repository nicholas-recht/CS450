import sys


class HardCoded:
    def train(self, attributes, targets, mapped):
        return

    def predict(self, attributes):

        predicts = []

        for row in attributes[0]:
            predicts.append(0)

        return predicts


class KNearestNeighbor:
    def __init__(self, k=1):
        self.k = k
        self.attributes = []
        self.targets = []
        self.mapped = []

    def train(self, attributes, targets, mapped):
        self.attributes = attributes
        self.targets = targets
        self.mapped = mapped

    def predict(self, test_attributes):
        # initialize data
        predictions = [-1] * len(test_attributes)

        for i, test_val in enumerate(test_attributes):
            distances = [float('inf')] * len(self.attributes)
            for j, train_val in enumerate(self.attributes):
                distances[j] = self.get_distance(test_val, train_val)
            k_nearest = self.get_k_nearest(distances)
            predictions[i] = self.get_avg_target(k_nearest)

        return predictions

    # gets euclidean distance between vector 1 and vector 2
    def get_distance(self, v1, v2):
        dist = 0.0
        for idx, val in enumerate(v1):
            # if it's not a mapped column find the distance in the standard way
            if not self.mapped[idx]:
                dist += (v1[idx] - v2[idx]) * (v1[idx] - v2[idx])
            # otherwise distance is 1 unless the values are the same
            else:
                dist += 0 if v1[idx] == v2[idx] else 1
        return dist

    # get the best prediction based of k nearest points
    def get_k_nearest(self, distances):
        k_nearest = [-1] * self.k
        k_distances = [sys.float_info.max] * self.k

        for idx, d in enumerate(distances):
            i = 0
            while i < self.k:
                if d < k_distances[i]:
                    k_distances[i] = d
                    k_nearest[i] = idx
                    i = self.k
                i += 1

        return k_nearest

    # predict a target based on k_nearest neighbors
    def get_avg_target(self, k_nearest):
        # all of our target values for k_nearest points
        targets = []
        for idx in k_nearest:
            targets.append(self.targets[idx])
        # sort the target values to easily find the mode
        targets = sorted(targets)

        longest_t = targets[0]    # currently longest t
        current_count = 0         # current count of t
        longest_count = 0         # longest count of t
        i = 1
        while i < len(k_nearest):
            t = targets[i]
            if t == targets[i - 1]:
                current_count += 1
            else:
                if current_count > longest_count:
                    longest_count = current_count
                    longest_t = targets[i - 1]
                    current_count = 0
            i += 1

        if current_count > longest_count:
            longest_t = targets[i - 1]

        return longest_t

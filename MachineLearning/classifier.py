import sys
import numpy as np
import copy
import random
import math


########################################################################################
# HardCoded
# A very simple classifier. Just returns 0 for every prediction.
########################################################################################
class HardCoded:
    def train(self, attributes, targets, mapped):
        return

    def predict(self, attributes):

        predicts = []

        for row in attributes[0]:
            predicts.append(0)

        return predicts


########################################################################################
# KNearestNeighbor
# Uses the k nearest neighbors algorithm to predict a class. The parameter k is the
# number of neighbors used for classification.
########################################################################################
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
                dist += 0 if v1[idx] == v2[idx] else .5
        return dist

    # get the best prediction based of k nearest points
    def get_k_nearest(self, distances):
        k_nearest = [-1] * self.k
        k_distances = [sys.float_info.max] * self.k

        for idx, d in enumerate(distances):
            i = 0
            while i < self.k:
                if d < k_distances[i]:
                    j = self.k - 1
                    while j >= i:
                        k_distances[j] = k_distances[j - 1]
                        k_nearest[j] = k_nearest[j - 1]
                        j -= 1
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


########################################################################################
# KNearestNeighbor
# Uses the k nearest neighbors algorithm to predict a class. The parameter k is the
# number of neighbors used for classification.
########################################################################################
class ID3:
    def __init__(self):
        self.attributes = []
        self.targets = []
        self.target_possible_values = []
        self.root = None

    def train(self, attributes, targets):
        self.attributes = attributes
        self.targets = targets

        # get the possible values of the target
        for row in self.targets:
            val = row[0]
            if val not in self.target_possible_values:
                self.target_possible_values.append(val)

        # create array of feature values
        features = [0] * len(attributes[0])
        for idx, val in enumerate(features):
            features[idx] = idx

        self.root = self.build_tree(attributes, targets, features)

    def build_tree(self, attributes, targets, features):
        """
        Recursive function to build a tree. Returns a leaf value containing the predicted target value, or
        a new ID3_Node
        :param attributes:
        :param targets:
        :param features:
        :return:
        """
        # get the number of various target values
        targets_map = {}
        for array_val in targets:
            val = array_val[0]
            if val in targets_map:
                targets_map[val] += 1
            else:
                targets_map[val] = 1

        largest_num = 0
        largest_target = -1
        for target, num in targets_map.items():
            if num > largest_num:
                largest_target = target

        # if all remaining share the same target value, then return it as a leaf
        if len(targets_map) <= 1:
            return targets[0][0]
        # if there are no more features remaining, return target value with most
        elif len(features) <= 0:
            return largest_target
        else:
            node = ID3Node()
            # assign the default value as the target with the most records
            node.default_value = largest_target

            # get the next feature to split on
            current_min = sys.float_info.max
            lowest_feature = -1
            for feature in features:
                feature_val = self.calc_info_gain(attributes, targets, feature)
                if feature_val < current_min:
                    current_min = feature_val
                    lowest_feature = feature

            # now separate targets and attributes
            node.attr_index = lowest_feature
            new_attributes = {}
            new_targets = {}
            for idx, attr_row in enumerate(attributes):
                val = attr_row[lowest_feature]
                if val not in new_attributes:
                    new_attributes[val] = []
                    new_targets[val] = []
                new_attributes[val].append(attributes[idx])
                new_targets[val].append(targets[idx])

            new_features = copy.deepcopy(features)
            new_features.remove(lowest_feature)

            # build the subtrees
            for key in new_attributes.keys():
                node.branches[key] = self.build_tree(new_attributes[key], new_targets[key], new_features)

            return node

    def predict(self, attributes):
        predicts = [None] * len(attributes)

        for idx, attribute in enumerate(attributes):
            predicts[idx] = self.traverse_tree(attribute, self.root)

        return predicts

    def traverse_tree(self, attr, node):
        # check and see if the branch exists
        if attr[node.attr_index] not in node.branches.keys():
            return node.default_value

        branch = node.branches[attr[node.attr_index]]

        # see if more nodes exist or else it's a leaf
        if isinstance(branch, ID3Node):
            return self.traverse_tree(attr, branch)
        else:
            return node.branches[attr[node.attr_index]]

    def output_tree(self, node, level):
        # create tabs
        tabs = ""
        i = 0
        while i < level:
            tabs += "\t"
            i += 1

        if isinstance(node, ID3Node):
            print(tabs, "node attribute: ", node.attr_index)
            print(tabs, "children: ")
            for key, val in node.branches.items():
                print(tabs, key, ": ")
                self.output_tree(val, level + 1)
        else:
            print(tabs, node)

    def calc_info_gain(self, attributes, targets, feature):
        """
        Calculates the total entropy value of the resulting set created by splitting the initial set
        by the given feature
        :param attributes:
        :param targets:
        :param feature:
        :return:
        """
        # get the count of each attribute val
        attribute_map = {}  # a map of the attribute value to a map of target values to a count of each
        attribute_totals = {} # a map of the attribute values of a total number of target values
        for att_index, attr in enumerate(attributes):
            val = attr[feature]
            if val not in attribute_map:
                attribute_map[val] = self.get_target_map()
                attribute_totals[val] = 0

            attribute_map[val][targets[att_index][0]] += 1
            attribute_totals[val] += 1

        # finally calculate the information gain
        gain = 0.0
        num_attributes = len(attributes)
        for attr, attr_num in attribute_totals.items():
            num_targets = attribute_totals[attr]
            entropy = 0.0
            for target_val, target_num in attribute_map[attr].items():
                entropy += self.calc_entropy(target_num / num_targets)
            # scale the value by the number of occurances of the attribute
            entropy *= attr_num / num_attributes

            gain += entropy

        return gain

    def calc_entropy(self, p):
        """
        Returns the entropy value for the given proportion or probability
        :param p:
        :return:
        """
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def get_target_map(self):
        """
        Returns a new map with all possible targets values as the keys and 0 as the value
        :return:
        """
        map = {}
        for val in self.target_possible_values:
            map[val] = 0

        return map


class ID3Node:
    def __init__(self):
        self.attr_index = 0
        self.branches = {}
        self.default_value = 0


########################################################################################
# Perceptron
# A neural network
########################################################################################
class Perceptron:
    def __init__(self, learning_rate=.1, nodes_per_layer=list()):
        self.node_layers = []
        self.num_targets = 0
        self.num_attributes = 0
        self.learning_rate = learning_rate
        self.bias = 1
        self.nodes_per_layer = nodes_per_layer

    def train(self, attributes, targets):
        self.num_targets = len(np.unique(targets.ravel()))
        self.num_attributes = len(attributes[0])

        # create the hidden layers
        for idx in range(len(self.nodes_per_layer)):
            if idx == 0:
                self.node_layers.append(NodeLayer(self.nodes_per_layer[idx],
                                                  self.num_attributes + 1, self.learning_rate))
            else:
                self.node_layers.append(NodeLayer(self.nodes_per_layer[idx],
                                                  self.nodes_per_layer[idx - 1] + 1, self.learning_rate))

        # create the output layer
        if len(self.node_layers) == 0:
            # if this is the only layer then the number of inputs is equal to the number of attributes + 1
            self.node_layers.append(NodeLayer(self.num_targets, self.num_attributes + 1, self.learning_rate))
        else:
            # else the number of inputs is equal to the number of nodes in the last hidden layer + 1
            self.node_layers.append(NodeLayer(self.num_targets,
                                              self.nodes_per_layer[len(self.nodes_per_layer) - 1] + 1,
                                              self.learning_rate))

        # split into training and validation set
        # num_in_split = len(attributes) * 33 // 100
        # v_attributes = attributes[:num_in_split]
        # t_attributes = attributes[num_in_split:]
        # v_targets = targets[:num_in_split]
        # t_targets = targets[num_in_split:]

        # start the learning process
        for i in range(300):
            # randomize the order each iteration
            zipped = list(zip(attributes, targets))
            random.shuffle(zipped)
            attributes, targets = zip(*zipped)

            num_right = 0

            for idx, attr in enumerate(attributes):
                predict = self.feed_forward(attr)
                self.back_propagate(targets[idx][0])

                # did we get it right?
                if predict == targets[idx][0]:
                    num_right += 1

            print(str(num_right / len(attributes) * 100))

    def predict(self, attributes):
        predicts = [None] * len(attributes)
        for idx, val in enumerate(attributes):
            predicts[idx] = self.feed_forward(attributes[idx])

        return predicts

    def feed_forward(self, input_row):
        layer_output = []
        for idx in range(len(self.node_layers)):
            if idx == 0:
                layer_output = self.node_layers[idx]\
                    .process(np.append(input_row, [self.bias]))
            else:
                layer_output = self.node_layers[idx]\
                    .process(np.append(layer_output, [self.bias]))

        # return the 0 index of the arg sort because this will correspond to the maximum value
        # from the output layer
        val = np.argmax(layer_output)
        return val

    def back_propagate(self, target):
        # calculate the error on the layer first
        self.node_layers[len(self.node_layers) - 1].calc_outer(target)

        # calculate the error on the other layers
        idx = len(self.node_layers) - 2
        while idx >= 0:
            self.node_layers[idx].calc_hidden(self.node_layers[idx + 1])
            idx -= 1

        # now adjust the weights
        for node_layer in self.node_layers:
            node_layer.back_propagate()


class Node:
    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.ranf(num_inputs) - .5
        self.learning_rate = learning_rate
        self.output = 0.0
        self.error = 0.0
        self.last_weight_change = [0.0] * num_inputs
        self.weight_constant = .9

    def process(self, inputs):
        dot_product = np.dot(self.weights, inputs)

        self.output = 1.0 / (1.0 + math.exp(-dot_product))

        return self.output

    def adjust_weights(self, inputs):
        # adjust the weight for each inputs by the function:
        #   w(i,j) = w(i,j) - learning_rate * error_rate * output[i]
        for idx, val in enumerate(self.weights):
            weight_change = -(self.learning_rate * self.error * inputs[idx] +
                              self.weight_constant * self.last_weight_change[idx])
            self.weights[idx] += weight_change
            self.last_weight_change[idx] = weight_change

    def calc_outer_error(self, target):
        """
        Sets the error value for this node based on the given target value
        :param target:
        :return:
        """
        self.error = self.output * (1 - self.output) * (self.output - target)

    def calc_hidden_error(self, index, node_layer):
        """
        Sets the error value for this node based on the given set of nodes in the next node layer
        :param index:
        :param node_layer:
        :return:
        """
        weighted_sum = 0.0
        for node in node_layer.nodes:
            weighted_sum += node.error * node.weights[index]

        self.error = self.output * (1 - self.output) * weighted_sum


class NodeLayer:
    def __init__(self, num_nodes, num_inputs, learning_rate):
        self.nodes = [None] * num_nodes
        self.inputs = []
        # init each node
        for idx, val in enumerate(self.nodes):
            self.nodes[idx] = Node(num_inputs, learning_rate)

    def process(self, inputs):
        self.inputs = inputs  # save the inputs so they can be used for error calculation
        output = [node.process(inputs) for node in self.nodes]

        return output

    def calc_outer(self, target):
        for idx, node in enumerate(self.nodes):
            if idx == target:
                node.calc_outer_error(1)
            else:
                node.calc_outer_error(0)

    def calc_hidden(self, prev_node_layer):
        for idx, node in enumerate(self.nodes):
            node.calc_hidden_error(idx, prev_node_layer)

    def back_propagate(self):
        for node in self.nodes:
            node.adjust_weights(self.inputs)

import csv
import numpy
import random


class DataSet:
    def __init__(self, file_path, ordered={}, target=-1, split=70):
        # load raw data
        reader = csv.reader(open(file_path, "rt"), delimiter=',')
        data_list = list(reader)

        # set target column value
        if target < 0:
            self.target_index = len(data_list[0]) - 1  # assume last position is targets
        else:
            self.swap_target(data_list, target)

            # swap the ordered parameter if necessary
            if target in ordered:
                tmp = ordered[target]
                ordered.pop(target, None)
                ordered[len(data_list[0]) - 1] = tmp
            self.target_index = len(data_list[0]) - 1

        # set ordered columns array
        self.ordered_columns = [False] * len(data_list[0])
        self.orderings = ordered
        for key in ordered.keys():
            self.ordered_columns[key] = True

        # set mapped columns array
        self.mapped_columns = [False] * len(data_list[0])
        self.mappings_from_str = [None] * len(data_list[0])
        self.mappings_to_str = [None] * len(data_list[0])
        for idx, val in enumerate(data_list[0]):
            if self.ordered_columns[idx] or not self.is_number(data_list[0][idx]):
                self.mapped_columns[idx] = True
                self.map_column(data_list, idx)

        # get attributes and targets
        random.shuffle(data_list)
        self.data_array = numpy.array(data_list).astype('float')
        self.attributes = self.data_array[:, :self.target_index]
        self.targets = self.data_array[:, self.target_index:self.target_index + 1]

        # normalize data
        self.data_normal = self.normalize(numpy.copy(self.data_array))
        self.attributes_normal = self.data_normal[:, :self.target_index]
        self.targets_normal = self.data_normal[:, self.target_index:self.target_index + 1]

        # split into training and testing set
        num_in_split = len(self.data_array) * split // 100
        self.train_attributes = self.attributes_normal[:num_in_split]
        self.test_attributes = self.attributes_normal[num_in_split:]
        self.train_targets = self.targets_normal[:num_in_split]
        self.test_targets = self.targets_normal[num_in_split:]

    # map the values of the column to integer values
    def map_column(self, data_list, column):
        # init values
        self.mappings_from_str[column] = {}
        self.mappings_to_str[column] = {}
        if not self.ordered_columns[column]:
            # get unique column values
            unique_values = set()
            for idx, val in enumerate(data_list):
                unique_values.add(data_list[idx][column])

            i = 0
            for val in unique_values:
                self.mappings_from_str[column][val] = i
                self.mappings_to_str[column][i] = val
                i += 1
        else:
            # use the ordering specified
            i = 0
            for val in self.orderings[column]:
                self.mappings_from_str[column][val] = i
                self.mappings_to_str[column][i] = val
                i += 1

        # replace column values
        for idx, val in enumerate(data_list):
            data_list[idx][column] = self.mappings_from_str[column][data_list[idx][column]]

    # swap the target column with the end column
    def swap_target(self, data_list, target):
        end = len(data_list[0]) - 1
        for idx, val in enumerate(data_list):
            # swap the values of the target column and end column
            tmp = data_list[idx][end]
            data_list[idx][end] = data_list[idx][target]
            data_list[idx][target] = tmp

    def normalize(self, data):
        # transpose the data so we can work with rows
        data = data.transpose()

        # use z scores for each row
        for idx, val in enumerate(data):
            if not self.mapped_columns[idx]:
                data[idx] = (data[idx] - data[idx].mean()) / data[idx].std()

        return data.transpose()

    # returns if s is a number
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

import csv
import numpy
import random


########################################################################################
# DataSet class
# A class representing an entire set of data to be used for machine learning.
#
# Parameters:
#   - file_path - path to csv file to open (',' used as delimiter)
#   - ordered - a map of column number to specified ordering for non-numerical data
#       * e.g. { 1: ['low','medium','high'] }
#       * the specified orderings will then be used for replacing values in the data
#   - target - column number of the target (default is last column)
#   - split - percentage of data to use for training (1 - 100)
#   - ignore - array of columns in the data set to ignore
#
# Members:
#   - target_index - index of target column in data set
#   - ordered_columns - array of boolean values. True when columns were given a
#                       specified order by the user. False otherwise.
#   - mapped_columns - array of boolean values. True when columns were originally
#                       unordered, non-numerical data, which values have now been
#                       replaced with integer values. False otherwise.
#   - mappings_from_str - map of strings values to corresponding integer placeholder
#                         for each column. Applies to both mapped and ordered columns.
#   - mappings_to_str - map of integer values to corresponding original string values
#                       for each column. Applies to both mapped and ordered columns.
#   - data_array - the entire collection of data (including attributes and targets)
#                  in numerical form.
#   - attributes - the attributes of the data
#   - targets - the targets of the data
#   - data_normal - the entire collection of data after being normalized using z-scores.
#                   Only mapped columns (unordered) are not normalized.
#   - attributes_normal - the normalized attributes
#   - targets_normal - the normalized target values
#   - train_attributes - the normalized attributes in the training set
#   - test_attributes - the normalized attributes in the testing set
#   - train_targets - the normalized targets in the training set
#   - test_targets - the normalized targets in the testing set
########################################################################################
class DataSet:
    def __init__(self, file_path, ordered=None, target=-1, split=70, ignore=None, missing='?'):
        # load raw data
        reader = csv.reader(open(file_path, "rt"), delimiter=',')
        data_list = list(reader)

        # replace any missing values
        for iRow, row in enumerate(data_list):
            for iCol, val in enumerate(row):
                if val == missing:
                    data_list[iRow][iCol] = 0

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
                if not self.ordered_columns[idx]:
                    self.mapped_columns[idx] = True
                self.map_column(data_list, idx)

        # get attributes and targets
        random.shuffle(data_list)
        self.data_array = numpy.array(data_list).astype('float')

        # "ignore" any given columns
        self.data_array = self.data_array.transpose()
        for col in ignore:
            self.data_array[col] = [1] * len(self.data_array[col])
        self.data_array = self.data_array.transpose()

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
            if not self.mapped_columns[idx] and data[idx].std() > 0:
                data[idx] = (data[idx] - data[idx].mean()) / data[idx].std()

        return data.transpose()

    # returns if s is a number
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

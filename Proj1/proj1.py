import sys
import getopt
from sklearn import datasets
import random
import datetime
import classifier


def main(argv):
    usage = "\tusage: dataset=[file_name] split=[1-100] classifier=[algorithm_name]"

    # get args
    opts, args = getopt.getopt(argv, "h", ["help", "dataset=", "split=", "classifier="])
    # handle args
    dataSet = className = "none"
    split = 30

    for key, val in opts:
        if key in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif key == "--dataset":
            dataSet = val
        elif key == "--split":
            split = int(val)
        elif key == "--classifier":
            className = val
        else:
            assert False, "unhandled option"

    # Load the data set
    attributes = targets = targetNames = None

    # default load iris data
    if dataSet == "none":
        attributes = datasets.load_iris().data
        targets = datasets.load_iris().target
        targetNames = datasets.load_iris().target_names
    # load from a csv file
    else:
        print("Do stuff")

    # Randomize the Order
    # thanks to http://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order for this method
    time = int((datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds())
    random.seed(time)
    z = list(zip(attributes, targets))
    random.shuffle(z)
    attributes, targets = zip(*z)

    # split into training and testing set
    numInSplit = len(attributes) * split // 100
    trAttributes = attributes[:numInSplit]
    teAttributes = attributes[numInSplit:]
    trTargets = targets[:numInSplit]
    teTargets = targets[numInSplit:]

    # select the network to use
    net = None
    if className == "HardCoded":
        net = classifier.HardCoded()
    else:
        net = classifier.HardCoded()

    # train the network
    net.train((trAttributes, trTargets))
    # predict
    predictions = net.predict((teAttributes, teTargets))

    # test the predictions
    i = 0
    numRight = 0
    while i < len(predictions):
        if predictions[i] == teTargets[i]:
            numRight += 1
        i += 1

    print("The number of correct predictions is: ", str(numRight / len(predictions) * 100), "%")

if __name__ == "__main__":
    main(sys.argv)

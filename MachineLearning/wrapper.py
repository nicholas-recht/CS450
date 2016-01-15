import sys
import getopt
import classifier
import preprocessor


def main(argv):
    usage = "\tusage: dataset=[file_name] split=[1-100] classifier=[algorithm_name] target=[index] \
            ordered=[column:val0|val1|val2,column2:val0|val1],etc"

    # get args
    opts, args = getopt.getopt(argv[1:], "h", ["help", "dataset=", "split=", "classifier=", "target=", "ordered="])
    # handle args
    data_set = class_name = "none"
    split = 70
    target = -1
    ordered = {}

    for key, val in opts:
        if key in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif key == "--dataset":
            data_set = val
        elif key == "--split":
            split = int(val)
        elif key == "--classifier":
            class_name = val
        elif key == "--target":
            target = int(val)
        elif key == "--ordered":
            entries = val.split(",")
            for entry in entries:
                entry_split = entry.split(":")
                ordered[int(entry_split[0])] = entry_split[1].split("|")
        else:
            assert False, "unhandled option"

    # Load the data set
    data = None

    # default load iris data
    if data_set == "none":
        print("No data set specified")
        sys.exit()
    # load from a csv file
    else:
        data = preprocessor.DataSet(data_set, ordered=ordered, target=target, split=split)

    # select the network to use
    net = None
    if class_name == "HardCoded":
        net = classifier.HardCoded()
    elif class_name == "KNearestNeighbors":
        net = classifier.KNearestNeighbor(k=5)
    else:
        net = classifier.HardCoded()

    # train the network
    net.train(data.train_attributes, data.train_targets, data.mapped_columns)
    # predict
    predictions = net.predict(data.test_attributes)

    # test the predictions
    i = 0
    num_right = 0
    while i < len(predictions):
        if predictions[i] == data.test_targets[i]:
            num_right += 1
        i += 1

    print("The number of correct predictions is: ", str(num_right / len(predictions) * 100), "%")

if __name__ == "__main__":
    main(sys.argv)

import sys
import getopt
import classifier
import preprocessor
from sklearn.neighbors import KNeighborsClassifier


def main(argv):
    usage = "\tusage: --dataset=[file_name] --split=[1-100] --classifier=[algorithm_name] --target=[index] \
            --ordered=[column:val0|val1|val2,column2:val0|val1],etc --ignore=[col1,col2,etc] \
            --mapped=[1,2,3,etc] --normalize"

    # get args
    opts, args = getopt.getopt(argv[1:], "h", ["help", "dataset=", "split=", "classifier=", "target=", "ordered=",
                                               "ignore=", "mapped=", "normalize"])
    # handle args
    data_set = class_name = "none"
    split = 70
    target = -1
    ordered = {}
    ignore = []
    mapped = []
    norm = False

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
        elif key == "--ignore":
            ignore = val.split(",")
            for idx, s in enumerate(ignore):
                ignore[idx] = int(s)
        elif key == "--ordered":
            entries = val.split(",")
            for entry in entries:
                entry_split = entry.split(":")
                ordered[int(entry_split[0])] = entry_split[1].split("|")
        elif key == "--mapped":
            entries = val.split(",")
            for entry in entries:
                mapped.append(int(entry))
        elif key == "--normalize":
            norm = True

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
        data = preprocessor.DataSet(data_set, ordered=ordered, target=target, split=split, ignore=ignore,
                                    mapped=mapped, norm=norm)

    # select the network to use
    net = None
    predict_targets = None

    if class_name == "HardCoded":
        net = classifier.HardCoded()
        # train
        net.train(data.train_attributes, data.train_targets, data.mapped_columns)
        # predict
        predictions = net.predict(data.test_attributes)
        predict_targets = data.test_targets
    elif class_name == "KNearestNeighbors":
        net = classifier.KNearestNeighbor(k=1)
        # train
        net.train(data.train_attributes, data.train_targets, data.mapped_columns)
        # predict
        predictions = net.predict(data.test_attributes)
        predict_targets = data.test_targets
    elif class_name == "KNearestNeighbors_alt":
        net = KNeighborsClassifier(n_neighbors=5)
        # train
        net.fit(data.train_attributes, data.train_targets.ravel())
        # predict
        predictions = net.predict(data.test_attributes)
        predict_targets = data.test_targets
    elif class_name == "DecisionTree":
        net = classifier.ID3()
        # train
        net.train(data.train_attributes, data.train_targets)
        net.output_tree(net.root, 0)
        # predict
        predictions = net.predict(data.test_attributes)
        predict_targets = data.test_targets
    else:
        print("Unrecognized classifier")
        sys.exit(1)

    # test the predictions
    i = 0
    num_right = 0
    while i < len(predictions):
        if predictions[i] == predict_targets[i]:
            num_right += 1
        i += 1

    print("The number of correct predictions is: ", str(num_right / len(predictions) * 100), "%")

if __name__ == "__main__":
    main(sys.argv)

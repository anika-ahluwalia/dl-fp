import pandas as pd

def get_data(file_path, inputs_header, labels_header):
    dataset = pd.read_csv(file_path)
    inputs = dataset[inputs_header].to_list()
    labels = dataset[labels_header].to_list()
    print(labels)
    # we will split the dataset equally between training and testing
    split_index = len(inputs) // 2
    training_inputs = inputs[0:split_index + 1]
    training_labels = labels[0:split_index + 1]
    testing_inputs = inputs[split_index:]
    testing_labels = labels[split_index:]
    return training_inputs, training_labels, testing_inputs, testing_labels
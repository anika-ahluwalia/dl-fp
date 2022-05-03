from model import AnalysisModel
from preprocess import get_data


def main():
    file_path = 'data/IMDBDataset.csv'
    input_header = "review"
    label_header = "sentiment"

    training_inputs, training_labels, testing_inputs, testing_labels = get_data(file_path, input_header, label_header)
    model = AnalysisModel()

if __name__ == '__main__':
    main()
import argparse
import RestorationModel from ..model 

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_pretrained", action="store_true")
    args = parser.parse_args()
    return args


def train(model, training_images, training_labels):
    print("training")

def test(model, testing_images, testing_labels):
    print("testing")


def main(args):
    print("starting!!")
    model = RestorationModel()
    train(model, None, None)
    test(model, None, None)

if __name__ == "__main__":
    args = parseArguments()
    main(args)

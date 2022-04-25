import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_pretrained", action="store_true")
    args = parser.parse_args()
    return args

def main(args):
    print("starting!!")
    

if __name__ == "__main__":
    args = parseArguments()
    main(args)

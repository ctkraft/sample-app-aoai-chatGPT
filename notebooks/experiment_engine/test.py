import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str)
    args = parser.parse_args()
    print(args.test)
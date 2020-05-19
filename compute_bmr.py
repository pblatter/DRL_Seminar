import os
import argparse
import pandas as pd 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting argparser')
    parser.add_argument('--mode', type=int, help='Blackjack mode', required=True)
    parser.add_argument('--version', '-v', type=str, help='Model version', required=True)
    args = parser.parse_args()

    MODE = args.mode
    VERSION = args.version

    LOG_PATH = f'reward_logs/rewards_{VERSION}_{MODE}.txt'

    if not os.path.isfile(LOG_PATH):
        print(f'Version and mode combination does not exist.')
        exit(0)

    df = pd.read_csv(LOG_PATH, header=None, delimiter=',')
    f = lambda x: float(x[:-1])
    vals = df.iloc[:,1].apply(f)
    max_rev = max(vals)

    if max_rev > 0.95:
        print(f'0.9')
    else:
        print(max_rev)

   
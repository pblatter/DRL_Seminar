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
    
    titles = {
            1: "Basic",
            2: "Regular",
            3: "All the 2's",
            4: "Random"
    }


    
    print(df.iloc[:,1])
    print(df.iloc[0,1][:-1])

    f = lambda x: float(x[:-1])
    vals = df.iloc[:,1].apply(f)
    print(vals.shape)
    print(vals.head())

    plt.plot(vals)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title(f'{titles[MODE]}')
    plt.ylim((0,1))
    plt.savefig(f'./plots/{VERSION}_{MODE}.png')
    #plt.show()
import pandas as pd
import numpy as np
from pathlib import Path

#load csv
def load_csv(dir, filename):
    file = dir/filename
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    
    df = pd.read_csv(file)

    #display file info
    print(f"Loaded {filename}: length: {len(df)} \n")

    return df

#save processed data
def save_csv(df, dir, filename):
    file = dir/filename
    df.to_csv(file, index=False) #dont include row indices
    print(f"Saved {filename}")

    return file

#split test sets
#arbitrary default values for each (used from book chap2)
def split_data(df, val=0.2, test=0.2, seed=2):
    n = len(df) #num rows in df

    #calculate rows for each set
    n_val = int(val*n)
    n_test = int(test*n)
    n_train = n - (n_val + n_test)

    #shuffle
    np.random.seed(seed) #fix random seed to get same results each time

    idx = np.arange(n) #create numpy array
    print('\nbefore shuffle', idx)

    np.random.shuffle(idx) #shuffle array
    print('\nafter shuffle', idx)

    df_shuffled = df.iloc[idx] #shuffled df

    #split into 3 sets
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val   = df_shuffled.iloc[n_train:n_train + n_val].copy()
    df_test  = df_shuffled.iloc[n_train + n_val:].copy()

    return df_train, df_val, df_test

#KMeans++: spreads initial centroids apart
def kmeans_plus_plus(X, k, random_state=1):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    centroids = []

    #1 pick first centroid randomly
    first_idx = np.random.randint(n_samples)
    centroids.append(X[first_idx])

    #2 compute the remaining k-1 centroids
    for i in range(1, k):
        #compute distances for all points from the nearest centroid
        distances = np.min(
            [np.linalg.norm(X - centroid, axis=1) ** 2 for centroid in centroids],
            axis=0
        )

        #normalize distances to get probabilities
        probs = distances / distances.sum()
        
        #choose the next centroid based on the computed probabilities
        next_idx = np.random.choice(n_samples, p=probs)
        centroids.append(X[next_idx])

    return np.array(centroids)
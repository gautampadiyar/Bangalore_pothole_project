from __future__ import division
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_samples, silhouette_score
from clean_data import load_csv, fil_empty, smooth, reset_isbump
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd
import progressbar
import time
import os


def apply_smoothing(df, window='flat', axis='accz'):
    x = df.index.values
    y = df[axis].values

    # windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    smoothed = smooth(y,5,window)
    df[axis] = pd.Series(smoothed, x)

    return df


def find_peaks(df, plot=False, in_timeseries=True):
    # code works for non time series data
    xs = df.index.values
    accz = df.accz.values
    peakind = signal.find_peaks_cwt(accz, np.array([1, 20]))

    inds = np.where(df.isbump == 1)
    new_peakind = np.intersect1d(inds, np.array(peakind))
    if new_peakind.size == 0:
        return None
    peaks = accz[new_peakind]
    ts = xs[new_peakind]

    if plot:
        if in_timeseries:
            plt.plot(xs, accz, 'b')
            plt.axhline(7.5, color='red', linestyle='-', linewidth=.5)
            plt.axhline(12.5, color='red', linestyle='-', linewidth=.5)
            plt.scatter(ts, peaks, color='red', marker='x', s=80)
            plt.show()

        else:
            plt.plot(xs, accz, 'b')
            plt.axhline(7.5, color='red', linestyle='-', linewidth=.5)
            plt.axhline(12.5, color='red', linestyle='-', linewidth=.5)
            plt.scatter(new_peakind, peaks, color='red', marker='x', s=80)
            plt.show()

    if in_timeseries:
        return df.loc[ts, :]
    else:
        return df.loc[new_peakind, :]

def compute_k_means(df): # compute single k-means clustering
    X = df[['latitude', 'longitude']].values
    n_clusters = 3
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    # KMeans

    # C = np.array([np.array(centroid) for centroid in k_means.clusters.keys()])
    labels = k_means.labels_

    plt.figure(figsize=(4, 3))

    plt.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float))

    plt.show()

def find_optimum_n_clusters(df, min_num_clusters, max_num_clusters):
    X = df[['latitude', 'longitude']].values

    range_n_clusters = range(min_num_clusters, max_num_clusters)
    score_list = []
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        print n_clusters
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, silhouette_avg)
        score_list.append(silhouette_avg)

    plt.plot(range_n_clusters, score_list)
    plt.show()



def generate_bump_df(data_folder):
    frames = []

    len_progress = len(os.listdir(data_folder)[1:])
    bar = progressbar.ProgressBar(maxval=len_progress, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    for idx, filename in enumerate(os.listdir(data_folder)[1:]):
        filepath = os.path.join(data_folder, filename)

        df = load_csv(filepath)
        df = fil_empty(df).set_index('timestamp').resample('13L').mean()
        # df = apply_smoothing(df, window='blackman')
        df = reset_isbump(df)
        temp_df = find_peaks(df, in_timeseries=True)
        frames.append(temp_df)
        bar.update(idx + 1)

    result = pd.concat(frames, ignore_index=True)
    bar.finish()
    return result

def master():
    """

    """
    df = generate_bump_df('data')
    print df.shape
    compute_k_means(df)
    return df

if __name__ == '__main__':
    # df = generate_bump_df('data')
    compute_k_means(df)
    # df = load_csv('data/18Jan2017102015.csv')
    # df = load_csv('data/16Jan201745618PM.csv')
    # df = fil_empty(df)
    # df.set_index('timestamp', inplace=True)
    # df = df.resample('13L').mean()
    # df = apply_smoothing(df, window='blackman')
    # df = reset_isbump(df)
    # find_peaks(df, plot=True, in_timeseries=False)
    # df = master()
    # find_optimum_n_clusters(df, 80, 210)

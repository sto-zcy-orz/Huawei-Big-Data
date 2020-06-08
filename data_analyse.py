import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import numpy as np
import pandas as pd

image_file = './image/'

if __name__ == '__main__':
    data = pd.read_csv('features.csv')
    test = pd.read_csv('test_fea.csv')

    keys = data.keys()
    print(keys)
    tot = 1
    for i in keys:
        if (isinstance(data[i][0], str) == True): continue
        if (i != 'label'): continue
        plt.figure(tot)
        plt.subplot(211)
        mu = np.mean(data[i]/60/24/60)
        sigma = np.std(data[i]/60/24/60)
        n, bins, patch = plt.hist(data[i]/60/24/60, 50, normed=1, alpha=0.5)
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.xlabel(i)
        plt.ylabel('Probability')
        plt.title(i+':   '+'mu: '+str(mu)+'sigma:  '+str(sigma))
        print(np.max(data[i]/60/24/60), np.min(data[i]/60/24/60))
        """
        plt.subplot(212)
        mu1 = np.mean(test[i])
        sigma1 = np.std(test[i])
        data1 = (test[i] - mu1)/sigma1*sigma + mu
        mu1 = np.mean(data1)
        sigma1 = np.std(data1)
        n, bins, patch = plt.hist(data1, 50, normed=1, alpha=0.5)
        y = norm.pdf(bins, mu1, sigma1)
        plt.plot(bins, y, 'r--')
        plt.xlabel(i)
        plt.ylabel('Probability')
        plt.title(i + ':   ' + 'mu: ' + str(mu1) + 'sigma:  ' + str(sigma1))

        #plt.savefig(image_file+i+'.jpg')
        #tot += 1
        #break"""

    # 对输出文件
    ans = pd.read_csv('result.csv')
    yi = pd.to_datetime(ans['ETA'], infer_datetime_format=True)
    er = pd.to_datetime(ans['onboardDate'], infer_datetime_format=True)
    print(yi - er)
    data = (yi - er).dt.total_seconds()
    print(data/60/24/60)
    plt.figure(2)
    mu = np.mean(data / 60 / 24 / 60)
    sigma = np.std(data / 60 / 24 / 60)
    n, bins, patch = plt.hist(data / 60 / 24 / 60, 50, normed=1, alpha=0.5)
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel(i)
    plt.ylabel('Probability')
    plt.title(i + ':   ' + 'mu: ' + str(mu) + 'sigma:  ' + str(sigma))
    plt.show()
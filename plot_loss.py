from matplotlib import pyplot as plt
import pickle
import numpy as np

with open('data/loss_plot.pkl','rb') as f:
    data = pickle.load(f)
    x = np.arange(len(data))*100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,data,color = 'green')
    ax.set_xlabel("train batches")
    ax.set_ylabel("train loss")
    plt.show()

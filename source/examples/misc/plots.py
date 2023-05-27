import matplotlib
from matplotlib import pyplot as plt

def plot_estimated_MI_trainig(epochs, estimated_MI, true_mi):
    fig, ax = plt.subplots()

    fig.set_figheight(9)
    fig.set_figwidth(16)

    # Сетка.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title("Mutual information estimate while training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$ I(X,Y) $")
    
    ax.minorticks_on()

    ax.plot(epochs, estimated_MI, label="$ \\hat I(X,Y) $")
    ax.hlines(y=true_mi, xmin=min(epochs), xmax=max(epochs), color='red', label="$ I(X,Y) $")

    ax.legend(loc='upper left')

    plt.show();
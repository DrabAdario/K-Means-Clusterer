import sys
import matplotlib.pyplot as plt

COLORS = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "cyan", "magenta"]


def cluster2d(filename):
    x = []
    y = []
    col = []

    with open(filename) as file:
    
        for line in file:
            if line.startswith('%'):
                continue

            cluster, xCoord, yCoord = line.split()

            x.append(float(xCoord))
            y.append(float(yCoord))
            col.append(COLORS[int(cluster)])

    plt.scatter(x,y, c=col)
    plt.show()

if __name__ == "__main__":
    cluster2d(sys.argv[1])
import sys
import matplotlib.pyplot as plt

COLORS = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "cyan", "magenta"]

def data2d(filename):
    x = []
    y = []
    
    with open(filename) as file:
    
        for line in file:
            if line.startswith('%'):
                continue

            xCoord, yCoord = line.split()

            x.append(float(xCoord))
            y.append(float(yCoord))

    plt.scatter(x,y)
    plt.show()

if __name__ == "__main__":
    data2d(sys.argv[1])
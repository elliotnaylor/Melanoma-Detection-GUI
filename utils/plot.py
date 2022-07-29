
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def draw_image(image):
    plt.imshow(image)
    plt.show()
    return

def draw_comparison(x, y, thresh):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axhline(y=thresh)
    ax1.plot(x, 'k--', linewidth=0.5)
    ax1.set_title("Horizontal colour symmetry")
    ax2.axhline(y=thresh)
    ax2.plot(y, 'k--', linewidth=0.5)
    ax2.set_title("Vertical colour symmetry")
    plt.show()
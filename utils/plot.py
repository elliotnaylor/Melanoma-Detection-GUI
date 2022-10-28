
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.inspection import DecisionBoundaryDisplay

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

def draw_svm_boundries(pipeline, xdata, ydata):
    fig, ax = plt.subplots()

    x0, x1 = xdata[:, 0], xdata[:, 1]
        
    disp = DecisionBoundaryDisplay.from_estimator(
        pipeline,
        xdata,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax
    )

    ax.scatter(x0, x1, c=ydata, cmap=plt.cm.coolwarm, s=5, edgecolors='k', linewidth=0.5)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM with RBF Kernel for Colour Asymmetry Detection')
    ax.set_xlabel('Size of Skin Lesion (100 / sample_size * i)')
    ax.set_ylabel('Colour difference (3D euclidean distance)')
    plt.show()
from scipy.spatial import distance
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
import numpy as np

#  read and show original image
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.metrics import confusion_matrix


def set_diagram(title, x_label, y_label, scale='linear'):
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())


def set_plot(x, y, diagram_type='ro', label=""):
    plt.plot(x, y, diagram_type, label=label)


def show_plot():
    plt.legend()
    plt.show()


def plot_confusion_matrix(true_labels, est_labels, normalize=False, title='Confusion matrix'):
    # confustion matrix based on the scikit-learn 
    cm = confusion_matrix(true_labels, est_labels)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


image = io.imread('bird.png')
io.imshow(image)
io.show()

rows, cols = image.shape[0], image.shape[1]
orginial_image = image.reshape(rows * cols, 3)
ks = range(3, 20)
costs = []
for k in ks:
    image = orginial_image
    kmeans_cluster = cluster.KMeans(n_clusters=k)
    kmeans_cluster.fit(image)
    cluster_centers = kmeans_cluster.cluster_centers_
    labels = kmeans_cluster.labels_
    print("k: " + str(k))
    labels = labels.reshape(rows, cols)
    print('cluster_centers:', cluster_centers)
    print('labels :', labels)

    interia = kmeans_cluster.inertia_
    print("k:", k, " cost:", interia)
    costs.append(interia)
    # show decompressed image
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            image[i, j, :] = cluster_centers[labels[i, j], :]
    io.imsave('compressed_image-' + str(k) + '.png', image)
    io.imshow(image)
    io.show()

title = "Cost Diagram"
xlabel = "k"
ylabel = "cost"
set_diagram(title, xlabel, ylabel)
set_plot(ks, costs)
show_plot()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle


def create_confusion_matrix(pred_labels,true_labels,file_name=None):
    """
    pred_labesl : list of integer labels
    true_labels : list of integer labels
    file_name : None means don't save only display
    otherwise, give it a path to save "confusion-matrix-1.jpg"
    """
    
    # class labels
    class_labels = list(set(pred_labels))

    # creates confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = cm / np.sum(cm, axis=1, keepdims=True)

    
    fig, ax = plt.subplots()
    # im = ax.imshow(cm, cmap='Blues')
    im = ax.imshow(accuracy, cmap='Blues', vmin = 0, vmax = 1) # Use accuracy values and set vmin/vmax

    # Add accuracy values to the plot
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({accuracy[i, j]:.2f})", ha='center', va='center')

    # Set axis labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    # Add colorbar
    ax.figure.colorbar(im, ax=ax)

    # Display the plot
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

    return fig

if __name__ == "__main__":
    path = "./results/"
    labelsPath = path + "labels.txt"

    # plot confusion matrix

    true_labels = []
    pred_labels = []
    with open(labelsPath, 'r') as out:
        for line in out:
            a, b = line.strip().split()
            pred_labels.append(int(a))
            true_labels.append(int(b))
            
    create_confusion_matrix(pred_labels,true_labels,file_name="confusion-matrix-1.jpg")
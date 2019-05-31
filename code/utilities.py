from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def write_output(out, class_probs, y_pred, ids):
    for i in range(0, len(ids)):
        out.write(ids[i])
        if y_pred[i] == 1:
            out.write(" true ")
        else:
            out.write(" false ")
        out.write(str(class_probs[i][y_pred[i]]) + "\n")

def print_metrics(y, y_pred):
    print("accuracy:", accuracy_score(y,y_pred))
    print("f1:", f1_score(y,y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y,y_pred))
    print()

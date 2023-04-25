import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tensorflow import keras
from keras.utils import plot_model
## import get_data_set
from main import get_data_set
import visualkeras
from keras_visualizer import visualizer 

# HASH = 1682420650.9587703
HASH = 1682430771.7856107

def main():
    model = keras.models.load_model(f'model-{HASH}.h5')
    X_test, y_test  =  get_data_set('test/')
    
    results = model.evaluate(X_test, y_test)
    print('test loss, test acc:', results)
    predictions = model.predict(X_test)
    print(predictions)

    #AUC
    actual = y_test[:,1]
    predicted = predictions[:,1]
    
    
    predicted[predicted <= 0.5] = 0.
    predicted[predicted > 0.5] = 1.
    # fpr, tpr, thresholds = metrics.roc_curve(actual, predicted)
    # plot confusion matrix
    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    print(confusion_matrix)

    # ## plot confusion matrix
    # plt.imshow(confusion_matrix, cmap='binary', interpolation='None')
    # plt.show()

    disp =  metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap="binary", values_format="d")
    plt.savefig(f'out/confusion_matrix-{HASH}.pdf') 
    plt.cla()

    # plot_model(model, to_file=f'out/model-{HASH}.pdf', show_shapes=True)

    visualizer(model, file_format='pdf', view=True)
if __name__ == '__main__':
    main()
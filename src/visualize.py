import matplotlib.pyplot as plt

def plot_training_curve(history):
    plt.figure(figsize=(7,5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title("Neural Network Training Curve (MAE per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lr_vs_nn(y_test, lr_pred, nn_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, lr_pred, alpha=0.5, label='Linear Regression')
    plt.scatter(y_test, nn_pred, alpha=0.5, label='Neural Network')
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI — LR vs NN")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weight_evolution(lr_weights, nn_weights):
    plt.figure(figsize=(10,5))
    plt.plot(lr_weights, 'o-', label='Initial LR Weights')
    plt.plot(nn_weights, 'x--', label='Final NN Weights')
    plt.title("First-Layer Weights: Linear Regression → Neural Network")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(True)
    plt.show()

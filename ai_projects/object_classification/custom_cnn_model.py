import numpy as np

class CustomSimpleNN:
    def __init__(self, input_size, hidden_size, num_classes, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, num_classes) * 0.01
        self.bias2 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, x, y_true, output):
        m = y_true.shape[0]

        y_true_one_hot = np.zeros((m, output.shape[1]))
        y_true_one_hot[np.arange(m), y_true] = 1  # One-hot encoding

        d_z2 = output - y_true_one_hot
        d_weights2 = np.dot(self.a1.T, d_z2) / m
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True) / m

        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        d_weights1 = np.dot(x.T, d_z1) / m
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True) / m

        # Ağırlıkları güncelle
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss



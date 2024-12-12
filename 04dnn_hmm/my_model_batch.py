import numpy as np

class MyDNN:
    """
    다층 신경망 (Deep Neural Network) 클래스 (드롭아웃 포함)
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dropout_rate=0.5):
        """
        생성자: 신경망의 구조를 정의하고 가중치 및 편향을 초기화합니다.

        Parameters:
        - dim_in (int): 입력 특징량의 차원 수.
        - dim_hidden (int): 각 은닉층의 뉴런(노드) 수.
        - dim_out (int): 출력층의 뉴런 수 (클래스 개수).
        - num_layers (int): 전체 층의 수 (은닉층 포함).
        - dropout_rate (float): 드롭아웃 비율.
        """
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.weights, self.biases = [], []  # 가중치와 편향 리스트 초기화

        # Xavier 초기화
        layer_dims = [dim_in] + [dim_hidden] * (num_layers - 1) + [dim_out]
        for i in range(len(layer_dims) - 1):
            self.weights.append(
                np.random.normal(0, 1 / np.sqrt(layer_dims[i]), (layer_dims[i + 1], layer_dims[i]))
            )
            self.biases.append(np.zeros((layer_dims[i + 1], 1)))

    def forward(self, x, train_mode=True):
        """
        순방향 계산 (Forward Pass)

        Parameters:
        - x (numpy.ndarray): 입력 데이터 (D, T).
        - train_mode (bool): 학습 모드 여부.

        Returns:
        - activations (list): 각 층의 활성화 값.
        - z_values (list): 각 층의 z 값 (가중합 결과).
        """
        activations = [x]
        z_values = []

        for i in range(self.num_layers):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)

            if i < self.num_layers - 1:  # 은닉층에는 ReLU
                a = self.relu(z)
                if train_mode and self.dropout_rate > 0:
                    a = self.apply_dropout(a)
            else:  # 출력층에는 Softmax
                a = self.softmax(z)

            activations.append(a)

        return activations, z_values

    def backward(self, batch_data):
        """
        역전파 계산 (Backward Pass).

        Parameters:
        - batch_data (list of tuples): [(activations, z_values, y)] 형식의 배치 데이터.

        Returns:
        - batch_dW (list): 가중치의 그래디언트 리스트.
        - batch_db (list): 편향의 그래디언트 리스트.
        """
        # 그래디언트 누적을 위한 초기화
        batch_dW = [np.zeros_like(w) for w in self.weights]
        batch_db = [np.zeros_like(b) for b in self.biases]

        for activations, z_values, y in batch_data:
            m = y.shape[1]  # 샘플 수
            dz = activations[-1] - y  # 출력층의 오류

            for i in reversed(range(self.num_layers)):
                dW = np.dot(dz, activations[i].T) / m
                db = np.sum(dz, axis=1, keepdims=True) / m
                batch_dW[i] += dW
                batch_db[i] += db

                if i > 0:
                    dz = np.dot(self.weights[i].T, dz) * self.relu_derivative(z_values[i - 1])
        return batch_dW, batch_db

    def update_weights(self, batch_dW, batch_db, learning_rate):
        """
        가중치 및 편향 업데이트.

        Parameters:
        - batch_dW (list): 가중치의 그래디언트 리스트.
        - batch_db (list): 편향의 그래디언트 리스트.
        - learning_rate (float): 학습률.
        """
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * batch_dW[i]
            self.biases[i] -= learning_rate * batch_db[i]

    def relu(self, x):
        """ReLU 활성화 함수."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU 활성화 함수의 도함수."""
        return (x > 0).astype(float)

    def apply_dropout(self, a):
        """드롭아웃 적용."""
        # 난수 생성기를 호출 시점마다 다르게 유지
        np.random.seed()  # 시드를 초기화하여 다른 마스크 생성
        dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
        a *= dropout_mask
        a /= (1 - self.dropout_rate)
        return a

    def softmax(self, x):
        """소프트맥스 함수."""
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def cross_entropy(self, predictions, targets):
        """
        교차 엔트로피 손실 함수.

        Parameters:
        - predictions (numpy.ndarray): 출력 확률 (softmax 결과).
        - targets (numpy.ndarray): 실제 라벨 (one-hot 인코딩).

        Returns:
        - loss (float): 계산된 손실 값.
        """
        epsilon = 1e-9
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.sum(targets * np.log(predictions))
        return loss / targets.shape[1]

    def set_weights(self, weights, biases):
        """
        외부에서 로드된 가중치와 편향 설정.

        Parameters:
        - weights (list of np.ndarray): 각 층의 가중치 리스트.
        - biases (list of np.ndarray): 각 층의 편향 리스트.
        """
        if len(weights) != self.num_layers or len(biases) != self.num_layers:
            raise ValueError("Mismatch in the number of layers for weights and biases.")

        for i in range(self.num_layers):
            if weights[i].shape != self.weights[i].shape or biases[i].shape != self.biases[i].shape:
                raise ValueError(f"Shape mismatch in layer {i}.")
            self.weights[i] = np.array(weights[i])
            self.biases[i] = np.array(biases[i])
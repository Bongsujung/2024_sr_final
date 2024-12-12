import numpy as np
#import time



class LearningRateScheduler:
    """
    학습률 스케줄러 클래스.
    
    웜업 단계와 감쇠 단계를 통해 학습률을 동적으로 조정.
    """
    def __init__(self, initial_lr, warmup_steps=0, decay_steps=0, decay_rate=0.96, min_lr=1e-6):
        """
        생성자: 스케줄러 초기화.

        Parameters:
        - initial_lr (float): 초기 학습률.
        - warmup_steps (int): 웜업 단계의 총 스텝 수.
        - decay_steps (int): 감쇠 단계의 총 스텝 수.
        - decay_rate (float): 감쇠 단계에서 학습률 감소 비율.
        - min_lr (float): 학습률의 최소 값.
        """
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.current_step = 0  # 현재 학습 스텝을 기록

    def get_lr(self):
        """
        현재 학습 스텝에 해당하는 학습률 반환.

        Returns:
        - lr (float): 현재 학습률.
        """
        if self.current_step < self.warmup_steps:
            # 웜업 단계: 학습률을 선형 증가.
            lr = self.initial_lr * (self.current_step + 1) / self.warmup_steps
        elif self.current_step < self.warmup_steps + self.decay_steps:
            # 감쇠 단계: 학습률 감소.
            decay_factor = self.decay_rate ** ((self.current_step - self.warmup_steps) / self.decay_steps)
            lr = self.initial_lr * decay_factor
        else:
            # 감쇠 이후: 마지막 계산된 학습률을 유지
            lr = self.last_lr

        # 최소 학습률로 제한
        lr = max(lr, self.min_lr)
        self.last_lr = lr  # 현재 학습률을 저장
        return lr


    def step(self):
        """
        학습 스텝을 1 증가.
        """
        self.current_step += 1



class Optimizer:
    """
    다양한 옵티마이저 클래스: SGD, RMSProp, Adam.
    """
    def __init__(self, model, lr_scheduler, optimizer_type="adam", 
                 learning_rate=0.01, beta1=0.9, beta2=0.999, rho=0.9, epsilon=1e-8):
        """
        생성자: 옵티마이저 초기화.

        Parameters:
        - model (MyDNN): 학습할 모델 객체.
        - lr_scheduler (LearningRateScheduler): 학습률 스케줄러 객체.
        - optimizer_type (str): 옵티마이저 유형 ("sgd", "rmsprop", "adam").
        - learning_rate (float): 학습률 (SGD 및 RMSProp용).
        - beta1 (float): 1차 모멘텀의 기여도 비율 (Adam용).
        - beta2 (float): 2차 모멘텀의 기여도 비율 (Adam용).
        - rho (float): RMSProp의 지수 감소 비율.
        - epsilon (float): 안정성을 위한 작은 값.
        """
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.epsilon = epsilon
        self.t = 0  # 타임스텝

        # 상태 초기화
        self.m = [np.zeros_like(w) for w in model.weights]  # 1차 모멘텀 (Adam)
        self.v = [np.zeros_like(w) for w in model.weights]  # 2차 모멘텀 (Adam)
        self.r = [np.zeros_like(w) for w in model.weights]  # RMSProp의 지수 가중 평균

    def step(self, batch_dW, batch_db):
        """
        가중치 및 편향 업데이트.

        Parameters:
        - batch_dW (list of np.ndarray): 배치별 가중치 기울기 리스트.
        - batch_db (list of np.ndarray): 배치별 편향 기울기 리스트.
        """
        self.t += 1  # 타임스텝 증가
        lr = self.lr_scheduler.get_lr()  # 현재 학습률 가져오기

        for i in range(len(self.model.weights)):
            if self.optimizer_type == "adam":
                # 1차 및 2차 모멘텀 계산 (Adam)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * batch_dW[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (batch_dW[i] ** 2)

                # 바이어스 보정 적용
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 가중치 및 편향 업데이트
                self.model.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.model.biases[i] -= lr * batch_db[i]

            elif self.optimizer_type == "rmsprop":
                # RMSProp 계산
                self.r[i] = self.rho * self.r[i] + (1 - self.rho) * (batch_dW[i] ** 2)

                # 가중치 및 편향 업데이트
                self.model.weights[i] -= lr * batch_dW[i] / (np.sqrt(self.r[i]) + self.epsilon)
                self.model.biases[i] -= lr * batch_db[i]

            elif self.optimizer_type == "sgd":
                # SGD 계산
                self.model.weights[i] -= lr * batch_dW[i]
                self.model.biases[i] -= lr * batch_db[i]

            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # 학습률 스케줄러 스텝 업데이트
        self.lr_scheduler.step()


from optimizers import LearningRateScheduler, Optimizer  # Optimizer 클래스 업데이트 필요
import numpy as np
from my_model_batch import MyDNN
import os
import json
import sys
import time


def save_model(model, config, filepath, save_config=True):
    """
    학습된 모델의 가중치, 편향, 설정 저장.

    Parameters:
    - model (MyDNN): 학습된 모델 객체.
    - config (dict): 학습 설정값.
    - filepath (str): 저장할 파일 경로(확장자 제외).
    """
    print("Saving model...")

    # 가중치와 편향 리스트를 numpy object로 변환
    processed_weights = [np.array(w) for w in model.weights]
    processed_biases = [np.array(b) for b in model.biases]

    # 저장 경로
    weights_file = filepath + '_weights.npz'
    np.savez(weights_file, weights=np.array(processed_weights, dtype=object), biases=np.array(processed_biases, dtype=object))
    print(f"Model weights saved to {weights_file}")

    if save_config:
        # 설정 파일 저장
        config_file = filepath + '_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model config saved to {config_file}")

def train_dnn(train_dataset, dev_dataset, config, log_file_path):
    """
    DNN 학습 루프 (미니 배치 단위로 역전파 수행, 옵티마이저 및 학습률 스케줄러 적용).

    Parameters:
    - train_dataset: 학습 데이터셋 객체.
    - dev_dataset: 검증 데이터셋 객체.
    - config (dict): 학습 설정값.
    - log_file_path (str): 로그 파일 저장 경로.
    """
    import math
    print("Initializing model...")

    # 모델 초기화
    model = MyDNN(
        dim_in=config['dim_in'],
        dim_hidden=config['dim_hidden'],
        dim_out=config['dim_out'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )

    # 학습률 스케줄러 및 옵티마이저 초기화
    lr_scheduler = LearningRateScheduler(
        initial_lr=config['initial_learning_rate'],
        warmup_steps=config.get('warmup_steps', 0),
        decay_steps=config.get('decay_steps', 100),
        decay_rate=config.get('decay_rate', 0.96),
        min_lr=config.get('min_learning_rate', 1e-6)
    )
    optimizer = Optimizer(
        model=model,
        lr_scheduler=lr_scheduler,
        optimizer_type=config.get('optimizer_type', 'adam'),  # 'sgd', 'rmsprop', 'adam' 중 선택
        learning_rate=config['initial_learning_rate']
    )

    # 초기값 설정
    best_loss = float('inf')  # 검증 손실의 최저값
    early_stop_counter = 0   # Early Stopping을 위한 카운터

    # 로그 파일 준비
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w') as log_file:
        log_file.write("Training Log\n")
        log_file.write(f"Config: {json.dumps(config, indent=4)}\n")
        log_file.write("Epoch, Train Loss, Train Accuracy, Dev Loss, Dev Accuracy, Time\n")

        for epoch in range(config['max_num_epoch']):
            # 에포크별 초기값 설정
            train_loss, train_correct, train_total = 0, 0, 0
            dev_loss, dev_correct, dev_total = 0, 0, 0
            epoch_start_time = time.time()  # 에포크 시작 시간 측정

            # 에포크 시작 메시지
            sys.stdout.write(f"\nEpoch {epoch + 1}/{config['max_num_epoch']} [")
            sys.stdout.flush()

            total_batches = math.ceil(len(train_dataset) / config['batch_size'])  # 전체 배치 수 계산
            completed_batches = 0

            # ===== 학습 =====
            for features, labels in train_dataset.get_batch(config['batch_size']):
                # 배치별 초기값 설정
                batch_data = []  # 역전파를 위한 데이터를 저장할 리스트

                # 배치 내 각 샘플 순회
                for feature, label in zip(features, labels):
                    feature = feature.T  # 입력 데이터의 차원 변경 (D, T)
                    label = np.array(label)  # 라벨 데이터를 numpy 배열로 변환 (T,)

                    # 원-핫 인코딩
                    y_one_hot = np.eye(config['dim_out'])[label].T  # (dim_out, T)

                    # 순방향 계산
                    activations, z_values = model.forward(feature, train_mode=True)
                    predictions = activations[-1]  # 출력층의 결과
                        
                    # 손실 계산
                    sample_loss = model.cross_entropy(predictions, y_one_hot)
                    train_loss += sample_loss  # 배치 손실 누적
                    train_total += len(label)

                    # 정확도 계산
                    predicted_classes = np.argmax(predictions, axis=0)
                    train_correct += np.sum(predicted_classes == label)

                    # 역전파 데이터를 저장
                    batch_data.append((activations, z_values, y_one_hot))

                # 배치 단위 역전파 수행 및 가중치 업데이트
                batch_dW, batch_db = model.backward(batch_data)
                optimizer.step(batch_dW, batch_db)

                # 진행도 업데이트
                completed_batches += 1
                progress = min(int((completed_batches / total_batches) * 50), 50)  # 50칸 기준, 최대 50
                percentage = min((completed_batches / total_batches) * 100, 100.0)  # 백분율 계산, 최대 100%
                sys.stdout.write("\r" + f"Epoch {epoch + 1}/{config['max_num_epoch']} [")
                sys.stdout.write("=" * progress + " " * (50 - progress) + f"] {percentage:.1f}%")
                sys.stdout.flush()

            # 에포크별 평균 학습 손실 및 정확도 계산
            lr = lr_scheduler.get_lr()
            print(f"Batch {completed_batches}/{total_batches}, LR: {lr:.6f}")
            train_loss /= train_total
            train_accuracy = train_correct / train_total

            # ===== 검증 =====
            for features, labels in dev_dataset.get_batch(config['batch_size']):
                for feature, label in zip(features, labels):
                    feature = feature.T  # 입력 데이터의 차원 변경 (D, T)
                    label = np.array(label)  # 라벨 데이터를 numpy 배열로 변환 (T,)

                    # 원-핫 인코딩
                    y_one_hot = np.eye(config['dim_out'])[label].T  # (dim_out, T)

                    # 순방향 계산
                    activations, _ = model.forward(feature, train_mode=False)
                    predictions = activations[-1]

                    # 손실 계산
                    sample_loss = model.cross_entropy(predictions, y_one_hot)
                    dev_loss += sample_loss

                    # 정확도 계산
                    predicted_classes = np.argmax(predictions, axis=0)
                    dev_correct += np.sum(predicted_classes == label)
                    dev_total += len(label)

            # 에포크별 평균 검증 손실 및 정확도 계산
            # 에포크별 시간 기록
            dev_loss /= dev_total
            dev_accuracy = dev_correct / dev_total
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
          
            # 로그 기록
            log_file.write(f"{epoch + 1}, {train_loss:.6f}, {train_accuracy:.2%}, {dev_loss:.6f}, {dev_accuracy:.2%}, {epoch_duration:.2f}\n")
            print(f"\nEpoch {epoch + 1}/{config['max_num_epoch']} Summary: "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2%}, "
                  f"Dev Loss: {dev_loss:.6f}, Dev Acc: {dev_accuracy:.2%}, Time: {epoch_duration:.2f}s")

            # === 체크포인트 저장 ===
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"./exp/models/checkpoints/checkpoint_epoch_{epoch + 1}"
                if (epoch + 1) == 5: # config는 한번만 저장 
                    save_model(model, config, checkpoint_path, save_config=True)
                else:
                    checkpoint_path = f"./exp/models/checkpoints/checkpoint_epoch_{epoch + 1}"
                    save_model(model, config, checkpoint_path, save_config=False)
                    print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

            # Early Stopping 조건 확인
            if dev_loss < best_loss:
                best_loss = dev_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter > config['early_stop_threshold']:
                    print("Early stopping triggered.")
                    break

    # 모델 저장
    save_model(model, config, "./exp/models/best_model", save_config=True)
    return model

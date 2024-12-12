import numpy as np
from hmmfunc import MonoPhoneHMM
from my_dataset_batch import CustomSequenceDataset
from train_dnn_batch import train_dnn
import os


def load_hmm_info(hmm_file_path):
    """
    HMM 정보를 로드하고 필요한 매개변수를 반환.

    Parameters:
    - hmm_file_path (str): HMM 파일 경로.

    Returns:
    - int: 음소 수 * 상태 수 (출력 차원).
    """
    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmm_file_path)
    return hmm.num_phones * hmm.num_states  # 전체 상태 수 반환


def load_mean_std(mean_std_file):
    """
    특징 값의 평균 및 표준 편차 로드.

    Parameters:
    - mean_std_file (str): 평균 및 표준편차 파일 경로.

    Returns:
    - tuple: (feat_mean, feat_std) 배열.
    """
    if not os.path.exists(mean_std_file):
        raise FileNotFoundError(f"Mean/Std file not found: {mean_std_file}")
    
    with open(mean_std_file, mode='r') as f:
        lines = f.readlines()
        if len(lines) < 4:
            raise ValueError(f"Invalid file format in {mean_std_file}")

        feat_mean = np.array(lines[1].split(), dtype=np.float32)
        feat_std = np.array(lines[3].split(), dtype=np.float32)
    
    return feat_mean, feat_std


if __name__ == "__main__":
    # 파일 경로 및 기본 설정
    base_dir = '../01compute_features/mfcc_delta/'
    train_feat_scp = os.path.join(base_dir, 'train/feats.scp') #train_per_500, train_small , train
    train_label_file = './exp/data/train_large/alignment'  # train_per_500, train_small , train_large
    
    dev_feat_scp = os.path.join(base_dir, 'train_small/feats.scp')
    dev_label_file = './exp/data/train_small/alignment'
    mean_std_file = os.path.join(base_dir, 'train/mean_std.txt')# train_per_500, train_small
    hmm_file_path = '../03gmm_hmm/exp/model_3state_4mix/20.hmm'

    log_file_path = './exp/models/logs/logging.csv'

    # 스플라이스 설정
    splice = 3
    # 학습 설정
    batch_size = 16
    total_utt = 105829 # 17500 , 1050 , 105829
    max_num_epoch = 100
    totla_step = (total_utt / batch_size) * max_num_epoch

    try:
        # 평균 및 표준편차 로드
        feat_mean, feat_std = load_mean_std(mean_std_file)

        # HMM 정보 로드
        dim_out = load_hmm_info(hmm_file_path)  # 출력 차원: 음소 수 x 상태 수
        feat_dim = len(feat_mean)  # 입력 특징량의 차원 수
        dim_in = feat_dim * (2 * splice + 1)  # 스플라이스된 입력 차원 계산

        # 데이터셋 초기화
        train_dataset = CustomSequenceDataset(train_feat_scp, train_label_file, feat_mean, feat_std, splice=splice, shuffle=True)
        dev_dataset = CustomSequenceDataset(dev_feat_scp, dev_label_file, feat_mean, feat_std, splice=splice, shuffle=False)

       
        
        config = {
            'dim_in': dim_in,                        # 입력 차원 (스플라이스 포함)
            'dim_hidden': int(dim_in * 2),           # 은닉층 차원 , int(dim_in * 2)
            'dim_out': dim_out,                      # 출력 차원 (음소 수 * 상태 수)
            'num_layers': 5,                         # 은닉층 개수
            'dropout_rate': 0,                       # 드롭아웃 비율
            'batch_size': batch_size,                # 배치 크기
            'max_num_epoch': max_num_epoch,          # 최대 에포크 수
            'initial_learning_rate': 0.01,           # 초기 학습률
            'warmup_steps': int(totla_step * 0.05),  # 학습률 warmup 스텝 수, 전체 스텝 5% 
            'decay_steps': int(totla_step / 3),      # 학습률 decay 스텝 수, 전체 스텝 1/3
            'decay_rate': 0.96,                      # 학습률 감소 비율
            'min_learning_rate': 1e-04,              # 최소 학습률
            'early_stop_threshold': 7,               # Early stopping 조건
            'optimizer_type': 'sgd'                  # 옵티마이저 종류 ('adam', 'sgd', 'rmsprop' 등)
        }
        
        print("Training configuration:")
        print(config)

        # 모델 학습
        model = train_dnn(train_dataset, dev_dataset, config, log_file_path)

    except Exception as e:
        print(f"An error occurred: {e}")
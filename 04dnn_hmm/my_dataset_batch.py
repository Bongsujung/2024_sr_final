import numpy as np

class CustomSequenceDataset:
    """
    특징량과 라벨 데이터를 로드하고, 배치를 생성하는 데이터셋 클래스 (패딩 제거).
    """

    def __init__(self, feature_file, label_file, mean, std, splice=0, shuffle=True):
        """
        데이터셋 초기화.
        특징량과 라벨 데이터를 로드.

        Parameters:
        - feature_file (str): 특징량 파일 경로.
        - label_file (str): 라벨 파일 경로.
        - mean (numpy.ndarray): 특징량 정규화를 위한 평균 값.
        - std (numpy.ndarray): 특징량 정규화를 위한 표준편차 값.
        - splice (int): 특징량의 스플라이싱 범위 (0이면 스플라이싱 없음).
        - shuffle (bool): 배치 생성 시 데이터를 셔플할지 여부.
        """
        self.shuffle = shuffle  # 셔플 여부 저장
        self.features, self.labels, self.feature_lengths, self.label_lengths = self.load_data(
            feature_file, label_file, mean, std, splice
        )

    def load_data(self, feature_file, label_file, mean, std, splice):
        """
        특징량과 라벨 데이터를 파일에서 로드.

        Parameters:
        - feature_file (str): 특징량 파일 경로.
        - label_file (str): 라벨 파일 경로.
        - mean (numpy.ndarray): 특징량 정규화를 위한 평균 값.
        - std (numpy.ndarray): 특징량 정규화를 위한 표준편차 값.
        - splice (int): 특징량의 스플라이싱 범위.

        Returns:
        - features (list of numpy.ndarray): 각 발화의 특징량 배열.
        - labels (list of numpy.ndarray): 각 발화의 라벨 배열.
        - feature_lengths (list of int): 각 발화의 특징량 시퀀스 길이.
        - label_lengths (list of int): 각 발화의 라벨 시퀀스 길이.
        """
        features, labels, feature_lengths, label_lengths = [], [], [], []

        # 특징량 데이터 로드
        with open(feature_file, 'r') as f_feat:
            for line in f_feat:
                path = line.strip().split()[1]  # 파일 경로를 읽음
                feat = np.fromfile(path, dtype=np.float32).reshape(-1, len(mean))  # 파일에서 데이터 로드
                feat = (feat - mean) / std  # 정규화 처리
                spliced_feat = self.splice_features(feat, splice)  # 스플라이싱 처리
                features.append(spliced_feat)  # 처리된 특징량 추가
                feature_lengths.append(spliced_feat.shape[0])  # 시퀀스 길이 저장

        # 라벨 데이터 로드
        with open(label_file, 'r') as f_label:
            for line in f_label:
                label = np.array(list(map(int, line.strip().split()[1:])))  # 라벨 데이터 읽음
                labels.append(label)  # 라벨 추가
                label_lengths.append(len(label))  # 라벨 길이 저장

        return features, labels, feature_lengths, label_lengths

    def splice_features(self, feat, splice):
        """
        특징량 데이터에 스플라이싱 적용.

        Parameters:
        - feat (numpy.ndarray): 원본 특징량 배열.
        - splice (int): 전후 프레임을 결합할 범위.

        Returns:
        - spliced (numpy.ndarray): 스플라이싱된 특징량 배열.
        """
        spliced = []
        for i in range(len(feat)):
            frame = []
            for offset in range(-splice, splice + 1):  # 스플라이싱 범위만큼 앞뒤 프레임 결합
                idx = i + offset
                # 유효한 인덱스는 원본 데이터를 사용, 그렇지 않으면 0으로 채움
                frame.append(feat[idx] if 0 <= idx < len(feat) else np.zeros(feat.shape[1]))
            spliced.append(np.concatenate(frame))  # 결합된 프레임 추가
        return np.array(spliced)

    def get_batch(self, batch_size):
        """
        배치를 생성하여 반환.

        Parameters:
        - batch_size (int): 배치 크기.

        Yields:
        - (list of numpy.ndarray, list of numpy.ndarray): 배치로 묶인 특징량과 라벨 데이터.
        """
        indices = np.arange(len(self.features))  # 데이터 인덱스 생성
        if self.shuffle:
            np.random.shuffle(indices)  # 셔플 적용

        for start_idx in range(0, len(self.features), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            # 배치로 묶어서 반환 (리스트 형태로 유지)
            yield [self.features[i] for i in batch_indices], [self.labels[i] for i in batch_indices]

    def __len__(self):
        """
        데이터셋의 총 샘플 수 반환.

        Returns:
        - int: 데이터셋에 포함된 샘플 수.
        """
        return len(self.features)



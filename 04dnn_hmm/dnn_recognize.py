import numpy as np
from hmmfunc import MonoPhoneHMM
from my_model_batch import MyDNN
import json
import os
#import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_model(model_path, config_path):
    """
    저장된 모델의 가중치, 편향, 설정 로드.

    Parameters:
    - filepath (str): 저장 파일의 공통 경로 (확장자 제외).

    Returns:
    - model (MyDNN): 로드된 MyDNN 모델 객체.
    - config (dict): 로드된 학습 설정.
    """
    # 설정 파일 로드
    config_file = config_path + ".json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from {config_file}: {config}")

    # 가중치 및 편향 파일 로드
    weights_file = model_path + ".npz"
    data = np.load(weights_file, allow_pickle=True)
    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]
    print(f"Weights and biases loaded from {weights_file}")

    # MyDNN 모델 초기화
    model = MyDNN(
        dim_in=config['dim_in'],
        dim_hidden=config['dim_hidden'],
        dim_out=config['dim_out'],
        num_layers=config['num_layers']
    )

    print("가중치 차원:", [w.shape for w in weights])
    print("바이어스 차원:", [b.shape for b in biases])

    # 가중치 및 편향 설정
    model.set_weights(weights, biases)
    print(f"Model weights and biases successfully set.")

    return model, config

class ModelEvaluator:
    """
    학습된 모델을 사용하여 테스트 데이터를 평가하고 결과를 저장하는 클래스.
    """
    def __init__(self, hmm_file, count_file, mean_std_file, lexicon_file, phone_list_file, splice=3, insert_sil=True):
        """
        초기화 함수.
        """
        # HMM 초기화
        self.hmm = MonoPhoneHMM()
        self.hmm.load_hmm(hmm_file)

        # 평균 및 표준 편차 로드
        self.feat_mean, self.feat_std = self._load_mean_std(mean_std_file)

        # 음소 리스트 로드
        self.phone_list = self._load_phone_list(phone_list_file)

        # 사전 파일 로드
        self.lexicon = self._load_lexicon(lexicon_file, self.phone_list, insert_sil)

        self.splice = splice

        # Prior 로드
        with open(count_file, mode='r') as f:
            line = f.readline()
            count = np.array(list(map(float, line.split())), dtype=np.float32)
            self.prior = count / np.sum(count)
            #print(self.prior)

    def _load_mean_std(self, mean_std_file):
        """특징량 평균 및 표준 편차 로드."""
        with open(mean_std_file, mode='r') as f:
            lines = f.readlines()
            feat_mean = np.array(lines[1].split(), dtype=np.float32)
            feat_std = np.array(lines[3].split(), dtype=np.float32)
        return feat_mean, feat_std

    def _load_phone_list(self, phone_list_file):
        """음소 리스트 로드."""
        phone_list = []
        with open(phone_list_file, mode='r') as f:
            for line in f:
                phone = line.strip().split()[0]
                phone_list.append(phone)
        return phone_list

    def _load_lexicon(self, lexicon_file, phone_list, insert_sil):
        """단어-음소 사전 로드."""
        lexicon = []
        with open(lexicon_file, mode='r') as f:
            for line in f:
                word, *phones = line.strip().split()
                if insert_sil:
                    phones = [phone_list[0]] + phones + [phone_list[0]]
                lexicon.append({
                    'word': word,
                    'pron': phones,
                    'int': [phone_list.index(ph) for ph in phones if ph in phone_list]
                })
        return lexicon
    
    def _splice_features(self, feat, splice):
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
        
    def _process_utterance(self, args):
        """
        ProcessPoolExecutor에서 호출할 수 있도록 매개변수를 단일 인자로 전달하도록 수정.
        """
        model, utt_id, feat_file, feat_dim, frame_dim, feat_mean, feat_std, splice, prior, lexicon, hmm = args
        
        try:
            feat = np.fromfile(feat_file, dtype=np.float32).reshape(-1, feat_dim)

            # 특징량 크기 확인
            if feat.shape[1] != len(feat_mean):
                print(f"Skipping {utt_id}: Feature dimension mismatch.")
                return utt_id, None

            # 정규화
            feat = (feat - feat_mean) / feat_std

            # Splicing
            spliced_feat = self._splice_features(feat, splice)
       
            #print("feat shape:" , feat.shape)
            # 모델 추론
            activations, _ = model.forward(spliced_feat.T, train_mode = False)
            softmax_output = activations[-1]

            # softmax_output이 확률 분포인지 확인
            assert np.allclose(np.sum(softmax_output, axis=0), 1), "Softmax output is not a probability distribution."
            likelihood = np.log(softmax_output / (prior[:, np.newaxis] + 1e-9))

            # Prior 값 확인
            assert prior.ndim == 1, "Prior must be a 1D array."
            assert np.all(prior > 0), "Prior values must be positive."
            assert np.isclose(np.sum(prior), 1, atol=1e-5), "Prior must sum to 1 (or close)."

            likelihood = likelihood.T  # (frames, states)
            # 차원 확인
            assert softmax_output.shape[0] == prior.shape[0], (
                f"Shape mismatch: softmax_output states ({softmax_output.shape[0]}) != prior states ({prior.shape[0]})"
            )

            # likelihood가 유효한 값인지 확인
            assert not np.any(np.isnan(likelihood)), "NaN found in likelihood calculation."
            assert not np.any(np.isinf(likelihood)), "Infinity found in likelihood calculation."
            # HMM 디코딩
            result, _ = hmm.recognize_with_dnn(likelihood, lexicon)
            return utt_id, result
        except Exception as e:
            print(f"Error processing {utt_id}: {e}")
            return utt_id, None
        
    def _sort_and_save_results(self, results, result_file):
        """
        결과를 발화 ID(utt_id) 기준으로 정렬한 후 파일에 저장합니다.

        Parameters:
        - results (list of tuples): (utt_id, result) 형식의 결과 리스트.
        - result_file (str): 저장할 파일 경로.
        """
        # 발화 ID를 기준으로 정렬
        sorted_results = sorted(results, key=lambda x: x[0])  # utt_id 기준 정렬

        # 정렬된 결과 저장
        with open(result_file, mode='w') as rf:
            for utt_id, result in sorted_results:
                rf.write(f"{utt_id}\nResult = {result}\n")
        print(f"Sorted results saved to {result_file}")


    def evaluate(self, model, feat_scp, result_file):
        """
        ProcessPoolExecutor를 사용한 병렬 처리된 테스트 데이터 평가.
        """
        results = []

        # 발화 정보 수집
        utterances = []
        with open(feat_scp, mode='r') as f:
            for line in f:
                parts = line.strip().split()
                utt_id, feat_file, frame_dim, feat_dim = parts[0], parts[1], int(parts[2]), int(parts[3])
                utterances.append((model, utt_id, feat_file, feat_dim, frame_dim, 
                                   self.feat_mean, self.feat_std, self.splice, self.prior, self.lexicon, self.hmm))

        # 병렬 처리
        with ProcessPoolExecutor(max_workers=int(os.cpu_count()/4)) as executor:  # 워커 수 조정 가능
            future_to_utt = {executor.submit(self._process_utterance, args): args[1] for args in utterances}

            for future in as_completed(future_to_utt):
                utt_id = future_to_utt[future]
                try:
                    result = future.result()
                    if result[1] is not None:  # 디코딩 성공
                        results.append(result)
                except Exception as e:
                    print(f"Error in processing {utt_id}: {e}")

    
        # 결과를 정렬한 후 저장
        self._sort_and_save_results(results, result_file)




   
if __name__ == "__main__":
    model_path = "./exp/models/SOTA/best_model_weights"  
    config_path = "./exp/models/SOTA/best_model_config" 
    hmm_file = "./exp/models/SOTA/20.hmm"
    count_file = "./exp/models/SOTA/state_counts_large"
    mean_std_file = "../01compute_features/mfcc_delta/train/mean_std.txt"
    lexicon_file = "../03gmm_hmm/sc35.dic"
    phone_list_file = "../03gmm_hmm/exp/data/train_large/phone_list"
    feat_scp = "../01compute_features/mfcc_delta/test/feats.scp"
    result_file = "./exp/test/result_dnn.txt"
    splice = 3  #학습 당시와 똑같아야 차원 맞음 
    insert_sil = True

    model, config = load_model(model_path, config_path)
   
    evaluator = ModelEvaluator(
        hmm_file=hmm_file,
        count_file=count_file,
        mean_std_file=mean_std_file,
        lexicon_file=lexicon_file,
        phone_list_file=phone_list_file,
        splice=splice,
        insert_sil=insert_sil
    )

    evaluator.evaluate(model, feat_scp, result_file)
    
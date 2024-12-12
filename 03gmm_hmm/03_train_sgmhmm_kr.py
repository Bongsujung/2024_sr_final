# -*- coding: utf-8 -*-

# 
# 혼합 수 1(Single Gaussian Model)의 HMM을 학습합니다.
# 

# 병렬처리를 위해 구현된 hmmfunc_parallel.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc_parallel import MonoPhoneHMM

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import sys
import os

# 
# 메인 함수
# 
if __name__ == "__main__":

    # 학습용 HMM 파일
    base_hmm = './exp_val/model_3state_1mix/0.hmm'

    # 훈련 데이터 특징량 리스트 파일
   # '../01compute_features/mfcc/train_small/feats.scp'
    feat_scp = \
        '../01compute_features/mfcc/train_per_500/feats.scp'

    # 훈련 데이터의 라벨 파일
    label_file = \
        './exp/data/train_per_500/text_int'

    # 업데이트 횟수
    num_iter = 10

    # 학습에 사용할 발화 수
    # 실제로는 모든 발화를 사용하지만, 시간이 걸리기 때문에
    # 이 프로그램에서는 일부 발화만 사용
    num_utters = 17500 # 17500 # 1050 # 105829

    # 출력 디렉토리
    out_dir = os.path.dirname(base_hmm)

    # 
    # 처리 시작
    # 

    # 출력 디렉토리가 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # 학습 전 HMM을 읽어들임
    hmm.load_hmm(base_hmm)

    # 라벨 파일을 열어 발화 ID별
    # 라벨 정보를 얻음
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0열은 발화 ID
            utt = line.split()[0]
            # 1열 이후는 라벨
            lab = line.split()[1:]
            # 각 요소는 문자열로 읽히므로,
            # 정수 값으로 변환
            lab = np.int64(lab)
            # label_list에 등록
            label_list[utt] = lab

    # 특징량 리스트 파일을 열고,
    # 발화 ID별 특징량 파일 경로를 얻음
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 특징량 경로를 feat_list에 추가
        # 이때, 학습에 사용할 발화 수만큼만 추가
        # (全てのデータを学習に用いると時間がかかるため)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0열은 발화 ID
            utt = line.split()[0]
            # 1열은 파일 경로
            ff = line.split()[1]
            # 3열은 차원 수
            nd = int(line.split()[3])
            # 발화 ID가 label_에 없으면 에러
            if not utt in label_list:
                sys.stderr.write(\
                    '%s does not have label\n' % (utt))
                exit(1)
            # 차원 수가 HMM의 차원 수와 일치하지 않으면 에러
            if hmm.num_dims != nd:
                sys.stderr.write(\
                    '%s: unexpected # dims (%d)\n'\
                    % (utt, nd))
                exit(1)
            # feat_file에 등록
            feat_list[utt] = ff


    # 
    # 학습 처리
    # 
    # num_iter 횟수만큼 업데이트 반복
    for iter in range(num_iter):
        print('%d-th iterateion' % (iter+1))
        # 학습(1회 반복)
        hmm.train_parallel(feat_list, label_list, num_workers= int(os.cpu_count()/6) )
       
        # HMM 프로토타입을 JSON 형식으로 저장
        out_hmm = os.path.join(out_dir, 
                               '%d.hmm' % (iter+1))
        # 학습한 HMM을 저장
        hmm.save_hmm(out_hmm)
        print('saved model: %s' % (out_hmm))


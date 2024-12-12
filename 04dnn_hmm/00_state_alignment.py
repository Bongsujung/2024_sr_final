# -*- coding: utf-8 -*-

#
# 훈련 데이터와 개발 데이터의
# HMM 상태 레벨에서의 정렬을 추정합니다
#

# hmmfunc.py에서 MonoPhoneHMM 클래스를 가져옵니다
from hmmfunc import MonoPhoneHMM

# 숫자 계산을 위한 모듈(numpy)을 가져옵니다
import numpy as np

# os, sys 모듈을 가져옵니다
import sys
import os

#
# 메인 함수
#
if __name__ == "__main__":

    # HMM 파일
    hmm_file = '../03gmm_hmm/exp/model_3state_4mix/20.hmm'

    # 훈련 데이터 특징 리스트 파일
    train_feat_scp = \
        '../01compute_features/mfcc_delta/train_small/feats.scp'
    # 개발 데이터 특징 리스트 파일
    # dev_feat_scp = \
    #     '../01compute_features/mfcc/dev/feats.scp'

    # 훈련 데이터 레이블 파일
    train_label_file = \
        '../03gmm_hmm/exp/data/train_small/text_int'
    # # 개발 데이터 레이블 파일
    # dev_label_file = \
    #     '../03gmm_hmm/exp/data/dev/text_int'

    # 훈련 데이터 정렬 결과 출력 파일
    train_align_file = \
        './exp/data/train_small/alignment'
    # # 개발 데이터 정렬 결과 출력 파일
    # dev_align_file = \
    #     './exp/data/dev/alignment'

    #
    # 처리 시작
    #

    # MonoPhoneHMM 클래스 호출
    hmm = MonoPhoneHMM()

    # 학습 이전의 HMM 파일을 읽습니다
    hmm.load_hmm(hmm_file)

    # 훈련/개발 데이터의
    # 특징/레이블/정렬 파일
    # 을 리스트로 생성합니다
    feat_scp_list = [train_feat_scp]
    label_file_list = [train_label_file]
    align_file_list = [train_align_file]

    for feat_scp, label_file, align_file in \
          zip(feat_scp_list, label_file_list, align_file_list):

        # 출력 디렉토리
        out_dir = os.path.dirname(align_file)

        # 출력 디렉토리가 존재하지 않으면 생성합니다
        os.makedirs(out_dir, exist_ok=True)

        # 레이블 파일을 열고 발화 ID별
        # 레이블 정보를 얻습니다
        label_list = {}
        with open(label_file, mode='r') as f:
            for line in f:
                # 0번째 열은 발화 ID
                utt = line.split()[0]
                # 1번째 열 이후는 레이블
                lab = line.split()[1:]
                # 각 요소는 문자열로 읽히므로,
                # 정수로 변환합니다
                lab = np.int64(lab)
                # label_list에 등록
                label_list[utt] = lab

        # 발화별로 정렬을 추정합니다
        with open(align_file, mode='w') as fa, \
             open(feat_scp, mode='r') as fs:
            for line in fs:
                # 0번째 열은 발화 ID
                utt = line.split()[0]
                print(utt)
                # 1번째 열은 파일 경로
                ff = line.split()[1]
                # 3번째 열은 차원 수
                nd = int(line.split()[3])

                # 발화 ID가 label_list에 없으면 오류 발생
                if not utt in label_list:
                    sys.stderr.write(\
                        '%s에 대한 레이블이 없습니다\n' % (utt))
                    exit(1)
                # 차원 수가 HMM 차원 수와 일치하지 않으면 오류 발생
                if hmm.num_dims != nd:
                    sys.stderr.write(\
                        '%s: 예상 차원 수와 다릅니다 (%d)\n'\
                        % (utt, nd))
                    exit(1)

                # 레이블을 얻습니다
                label = label_list[utt]
                # 특징 파일을 엽니다
                feat = np.fromfile(ff, dtype=np.float32)
                # 프레임 수 x 차원 수 배열로 변형합니다
                feat = feat.reshape(-1, hmm.num_dims)
                
                # 정렬 실행
                alignment = hmm.state_alignment(feat, label)
                # alignment는 숫자 리스트이므로
                # 파일에 쓰기 위해 문자열로 변환합니다
                alignment = ' '.join(map(str, alignment))
                # 파일에 출력합니다
                fa.write('%s %s\n' % (utt, alignment))
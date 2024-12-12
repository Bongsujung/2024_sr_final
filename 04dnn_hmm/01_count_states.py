# -*- coding: utf-8 -*-

#
# 추정된 정렬 결과를 기반으로
# 각 HMM 상태의 출현 횟수를 계산합니다.
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

    # 훈련 데이터 정렬 파일
    align_file = \
        './exp/data/train_small/alignment'

    # 계산된 사전 확률 파일
    count_file = \
        './exp/model_dnn/state_counts_small'

    #
    # 처리 시작
    #

    # MonoPhoneHMM 클래스 호출
    hmm = MonoPhoneHMM()

    # 학습 이전의 HMM 파일을 읽습니다
    hmm.load_hmm(hmm_file)

    # HMM의 총 상태 수를 얻습니다
    num_states = hmm.num_phones * hmm.num_states

    # 출력 디렉토리
    out_dir = os.path.dirname(count_file)

    # 출력 디렉토리가 존재하지 않으면 생성합니다
    os.makedirs(out_dir, exist_ok=True)

    # 상태별 출현 횟수 카운터
    count = np.zeros(num_states, np.int64)

    # 정렬 파일을 엽니다
    with open(align_file, mode='r') as f:
        for line in f:
            # 0번째 열은 발화 ID
            utt = line.split()[0]
            # 1번째 열 이후는 정렬 정보
            ali = line.split()[1:]
            # 정렬 정보는 문자열로 읽히므로,
            # 정수로 변환합니다
            ali = np.int64(ali)
            # 상태를 하나씩 읽으며,
            # 해당 상태의 카운터를 1 증가시킵니다
            for a in ali:
                count[a] += 1

    # 카운트가 0인 항목은 1로 설정합니다
    # 이후 처리에서 0으로 나누는 것을 방지하기 위해
    count[count == 0] = 1

    # 카운트 결과를 출력합니다
    with open(count_file, mode='w') as f:
        # 벡터 count를 문자열로 변환
        count_str = ' '.join(map(str, count))
        f.write('%s\n' % (count_str))
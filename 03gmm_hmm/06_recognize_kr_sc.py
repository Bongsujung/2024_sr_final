from hmmfunc import MonoPhoneHMM
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# 개별 발화 인식을 수행하는 함수 정의
def recognize_utterance(hmm, utt, ff, nd, lexicon, nbest):
    # 차원 수가 HMM의 차원 수와 일치하지 않으면 에러 반환
    if hmm.num_dims != nd:
        return f"{utt}: unexpected # dims ({nd})\n"

    # 특징량 파일을 읽고 고립 단어 인식을 수행
    feat = np.fromfile(ff, dtype=np.float32).reshape(-1, hmm.num_dims)
    result, detail = hmm.recognize(feat, lexicon)

    # 결과 포맷을 지정하여 반환
    output = [f"{utt} {ff}", f"Result = {result}"]
    if nbest > 0:
        output.append("[Ranking]")
        output.extend(f"  {res['word']} {res['score']}" for res in detail[:nbest])
    return "\n".join(output) + "\n"


# 메인 함수
if __name__ == "__main__":
    hmm_file = './exp/model_3state_4mix/20.hmm'
    feat_scp = '../01compute_features/mfcc_delta/test/feats.scp'
    lexicon_file = 'sc35.dic'
    phone_list_file = './exp/data/train_per_500/phone_list'
    insert_sil = True
    nbest = 0
    out_dir = './exp/data/test'
    result_file = os.path.join(out_dir, 'result.txt')
    os.makedirs(out_dir, exist_ok=True)

    # 음소 리스트 파일을 열어 phone_list에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            phone_list.append(line.split()[0])

    # 사전 파일을 열어 단어와 음소열의 대응 리스트를 얻음
    lexicon = []
    with open(lexicon_file, mode='r') as f:
        for line in f:
            word = line.split()[0]
            phones = line.split()[1:]
            if insert_sil:
                phones = [phone_list[0]] + phones + [phone_list[0]]
            ph_int = [phone_list.index(ph) for ph in phones if ph in phone_list]
            lexicon.append({'word': word, 'pron': phones, 'int': ph_int})

    # MonoPhoneHMM 초기화
    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmm_file)

    # 병렬 처리를 통해 발화별로 고립 단어 인식을 수행
    with ProcessPoolExecutor() as executor:
        futures = []
        with open(feat_scp, mode='r') as f:
            for line in f:
                utt, ff, _, nd = line.split()
                futures.append(
                    executor.submit(recognize_utterance, hmm, utt, ff, int(nd), lexicon, nbest)
                )

        # 결과를 파일에 저장
        with open(result_file, mode='w') as f_out:
            for future in futures:
                f_out.write(future.result())
# 2024년 2학기 음성인식입문 기말 프로젝트 🎓

### 정봉수 - 김동희 팀

![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/gimdonghui/Documents/INU/4-2/Intro_to_ASR/test/data/image.png?version%3D1737274808231)

---

## 🚀 실행 방법

```bash
cd 04dnn_hmm
python3 dnn_recognize.py
python3 eval.py
```

---

## 🌟 DNN-HMM (Hybrid System) NumPy로만 구현하기

## BASE CODE : https://github.com/bjpublic/python_speech_recognition

---

### 📖 프로젝트 개요

본 프로젝트는 음성 인식 모델의 기본적인 구현을 목표로 하며, **Deep Neural Network (DNN)**와 **Hidden Markov Model (HMM)**을 결합한 하이브리드 시스템을 **NumPy**만을 사용하여 개발하였습니다.  
파이썬의 고수준 라이브러리 없이 순수 계산과 알고리즘으로 음성 인식 시스템을 구현함으로써, DNN-HMM 구조의 근본 원리를 학습하고 이해하는 데 초점을 맞췄습니다.

---

## 📊 결과 분석

```plaintext
============ Results Analysis ============
Test: ./exp/test/result_dnn.txt
True: ../data/label/test/test_label.txt
Accuracy: 92.38%
Hits: 485, Total: 525
==========================================
```

- **정확도 (Accuracy)**: 92.38%
- **테스트 데이터 개수**: 525개
- **예측 적중**: 485개

### 🏆 성능 요약

NumPy만을 활용하여 구현한 DNN-HMM 모델은 높은 정확도를 달성하였으며, 간단한 Fully Connected (FC) Layer 모델을 사용하였습니다.

---

## 🛠 주요 구현 내용

### 1️⃣ **DNN 구조**

- Fully Connected (FC) Layer만으로 구성된 간단한 DNN 설계
- 활성화 함수로 ReLU 및 Softmax를 적용
- Loss Function: Cross-Entropy 사용

### 2️⃣ **HMM 구조**

- Gaussian Mixture Model (GMM) 기반 HMM 설계
- Forward-Backward 알고리즘 구현
- `3state-4mix/20.hmm`최종 사용

### 3️⃣ **결합 (Hybrid System)**

- DNN에서 생성된 음향 모델 확률값을 HMM에서 사용하는 방식으로 결합

---

## 📁 파일 구조

```
.
├── 01compute_features
│   ├── 01_compute_mfcc_delta.py
│   ├── 01_compute_mfcc_kr_sc.py
│   ├── 02_compute_mean_std_kr.py
│   └── mfcc_delta
├── 03gmm_hmm
│   ├── 00_make_label_kr.py
│   ├── 01_make_proto_kr.py
│   ├── 02_init_hmm_kr.py
│   ├── 03_train_gmmhmm_kr.py
│   ├── 03_train_sgmhmm_kr.py
│   ├── 06_recognize_kr_sc.py
│   ├── cmu39.txt
│   ├── eval.py
│   ├── exp
│   ├── hmmfunc.py
│   ├── hmmfunc_parallel.py
│   └── sc35.dic
├── 04dnn_hmm
│   ├── 00_state_alignment.py
│   ├── 01_count_states.py
│   ├── dnn_recognize.py
│   ├── eval.py
│   ├── exp
│   ├── hmmfunc.py
│   ├── main_batch.py
│   ├── my_dataset_batch.py
│   ├── my_model_batch.py
│   ├── optimizers.py
│   └── train_dnn_batch.py
├── README.md
├── data
     └── label
```

---

## ✔️ 구현된 특징

- **Log-Mel Spectrum**
- **MFCC**
- **MFCC-delta-delta**

---

## ✔️ 구현된 모델

- **Gaussian Mixture Model (GMM)**
- **Hidden Markov Model (HMM)**
- **Fully Connected DNN (DNN)**

---

## 🔍 데이터셋

- **사용 데이터셋**: Google speech commend V2
- `./data/wav/` 경로에 저장하여 사용

---

## 📜 라이선스

본 프로젝트는 2024년 음성인식입문 수업의 기말 프로젝트로 제작되었습니다.

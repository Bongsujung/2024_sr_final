# -*- coding: utf-8 -*-

import wave
import numpy as np
import os
import sys

class FeatureExtractor():
    def __init__(self, 
                 sample_frequency=16000, 
                 frame_length=25, 
                 frame_shift=10, 
                 num_mel_bins=23, 
                 num_ceps=13, 
                 lifter_coef=22, 
                 low_frequency=20, 
                 high_frequency=8000, 
                 dither=1.0):
        self.sample_freq = sample_frequency
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lifter_coef = lifter_coef
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.dither_coef = dither
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2
        self.mel_filter_bank = self.MakeMelFilterBank()
        self.dct_matrix = self.MakeDCTMatrix()
        self.lifter = self.MakeLifter()

    def Herz2Mel(self, herz):
        return (1127.0 * np.log(1.0 + herz / 700))

    def MakeMelFilterBank(self):
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        mel_points = np.linspace(mel_low_freq, mel_high_freq, self.num_mel_bins+2)
        dim_spectrum = int(self.fft_size / 2) + 1
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            left_mel = mel_points[m]
            center_mel = mel_points[m+1]
            right_mel = mel_points[m+2]
            for n in range(dim_spectrum):
                freq = 1.0 * n * self.sample_freq/2 / dim_spectrum
                mel = self.Herz2Mel(freq)
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight
        return mel_filter_bank

    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        1프레임 분량의 파형 데이터를 추출하여 전처리하고,
        로그 파워값을 계산
        '''
        # waveform에서 1프레임 분량의 파형 추출
        window = waveform[start_index:start_index + self.frame_size].copy()

        # 디더링을 수행
        # (-dither_coef～dither_coef 사이 값을 난수로 추가)
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2*self.dither_coef) - self.dither_coef

        # 직류 성분을 제거
        window = window - np.mean(window)

        # 아래 처리를 실행하기 전에 파워 계산
        power = np.sum(window ** 2)
        # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
        if power < 1E-10:
            power = 1E-10
        # 로그를 취한다
        log_power = np.log(power)

        # 프리엠퍼시스(고주파 강조)
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window,np.array([1.0, -0.97]), mode='same')
        # numpy.convolve는 0번째 요소가 처리되지 않기
        # (window[i-1]가 없기) 때문에
        # window[0-1]을window[0]로 대체하여 처리
        window[0] -= 0.97*window[0]

        # 해밍 창을 적용
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power

    def ComputeFBANK(self, waveform):
        num_samples = np.size(waveform)
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        log_power = np.zeros(num_frames)
        for frame in range(num_frames):
            start_index = frame * self.frame_shift
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)
            spectrum = np.fft.fft(window, n=self.fft_size)[:int(self.fft_size / 2) + 1]
            spectrum = np.abs(spectrum) ** 2
            fbank = np.dot(spectrum, self.mel_filter_bank.T)
            fbank[fbank < 0.1] = 0.1
            fbank_features[frame] = np.log(fbank)
            log_power[frame] = log_pow
        return fbank_features, log_power

    def MakeDCTMatrix(self):
        N = self.num_mel_bins
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) * np.cos(((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N))
        return dct_matrix

    def MakeLifter(self):
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        return 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)

    def ComputeMFCC(self, waveform):
        fbank, log_power = self.ComputeFBANK(waveform)
        mfcc = np.dot(fbank, self.dct_matrix.T)
        mfcc *= self.lifter
        mfcc[:, 0] = log_power
        return mfcc

    def ComputeDelta(self, features, N=2):
        """
        ComputeDelta 함수: 특징(feature) 데이터의 시간 차분(delta)를 계산합니다.

        Args:
            features (numpy.ndarray): 입력 특징 벡터 (frame x feature matrix).
            N (int): 시간차(window 크기)를 정의하는 정수. 기본값은 2.

        Returns:
            numpy.ndarray: 입력 특징에 대해 계산된 delta (frame x feature matrix).
        """
        # 특징 벡터의 프레임 수와 각 프레임의 특징 개수를 가져옵니다.
        num_frames, num_features = features.shape
        
        # Delta 계산을 위한 분모를 구합니다 (시간 차의 제곱합의 두 배).
        denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
        
        # Delta 특징 벡터를 저장할 배열을 입력 특징과 동일한 크기로 초기화합니다.
        delta_features = np.zeros_like(features)
        
        # 입력 특징을 양 끝으로 N만큼 확장(padding)하여 경계값 처리를 용이하게 합니다.
        # 패딩된 영역은 가장자리에 있는 값을 그대로 복제합니다.
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        
        # 각 프레임에 대해 Delta를 계산합니다.
        for t in range(num_frames):
            # Delta 계산: [-N, ..., 0, ..., +N]의 가중치를 적용하여 시간 차분 계산.
            delta_features[t] = np.dot(
                np.arange(-N, N + 1),         # Delta 가중치 벡터.
                padded[t:t + 2 * N + 1]      # 현재 프레임을 중심으로 한 윈도우.
            ) / denominator                  # 계산된 값을 분모로 나누어 정규화.
        
        return delta_features

    def ComputeFeaturesWithDelta(self, waveform):
        """
        ComputeFeaturesWithDelta 함수: 입력 신호의 MFCC와 Delta, Delta-Delta 특징을 계산하고 결합합니다.

        Args:
            waveform (numpy.ndarray): 입력 오디오 신호 (waveform).

        Returns:
            numpy.ndarray: 결합된 특징 벡터 (MFCC, Delta, Delta-Delta).
        """
        # MFCC를 계산합니다.
        mfcc = self.ComputeMFCC(waveform)
        
        # MFCC의 Delta (1차 시간차분)를 계산합니다.
        delta = self.ComputeDelta(mfcc)
        
        # Delta의 Delta (2차 시간차분)를 계산합니다.
        delta_delta = self.ComputeDelta(delta)
        
        # MFCC, Delta, Delta-Delta를 가로로 결합하여 최종 특징 벡터를 생성합니다.
        combined_features = np.hstack((mfcc, delta, delta_delta))
        
        return combined_features
        
if __name__ == "__main__":
    # 설정 시작

    # 각 wav 파일의 리스트와 특징량 출력 위치
    train_wav_scp = '../data/label/train/per_500_wav.scp'
    train_out_dir = './mfcc_delta/train_per_500'
    #dev_wav_scp = '../data/label/dev/wav.scp'
    #dev_out_dir = './mfcc/dev'
    test_wav_scp = '../data/label/test/wav.scp'
    test_out_dir = './mfcc_delta/test'

    # 샘플링 주파수 [Hz]
    sample_frequency = 16000
    # 프레임 길이 [밀리초]
    frame_length = 25
    # 프레임 시프트 [밀리초]
    frame_shift = 10
    
    # 저주파수 대역 제거의 컷오프 주파수 [Hz]
    low_frequency = 20
    # 고주파수 대역 제거의 컷오프 주파수 [Hz]
    high_frequency = sample_frequency / 2
    # 로그 Mel 필터뱅크 차원수
    num_mel_bins = 23
    # MFCC 차원수
    num_ceps = 13
    # 디더링 계수
    dither = 1.0

    # 난수 시드 설정(디더링 처리 결과의 재현성 확보)
    np.random.seed(seed=0)

    # 특징값 추출 클래스 불러오기
    feat_extractor = FeatureExtractor(
        sample_frequency=sample_frequency, 
        frame_length=frame_length, 
        frame_shift=frame_shift, 
        num_mel_bins=num_mel_bins, 
        num_ceps=num_ceps,
        low_frequency=low_frequency, 
        high_frequency=high_frequency, 
        dither=dither
    )

    # wav 파일 목록과 출력 위치를 리스트로 생성
    wav_scp_list = [train_wav_scp, 
                    #dev_wav_scp, 
                    test_wav_scp]
    out_dir_list = [train_out_dir, 
                    #dev_out_dir, 
                    test_out_dir]

    # 각 세트에 대한 처리 실행
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('Input wav_scp: %s' % (wav_scp))
        print('Output directory: %s' % (out_dir))

        # 특징값 파일 경로, 프레임 수, 차원수를 기록한 리스트
        feat_scp = os.path.join(out_dir, 'feats.scp')

        # 출력 디렉토리가 존재하지 않을 경우 생성
        os.makedirs(out_dir, exist_ok=True)

        # wav 목록 읽기 모드
        # 특징값 리스트를 쓰기 모드로 열기
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wav 리스트를 한 줄씩 읽음
            for line in file_wav:
                # 각 행에는 발화 ID와 wav 파일 경로가 스페이스로 구분되어 있으므로
                # split 함수를 써서 스페이스 구분 행을 리스트형 변수로 변환
                parts = line.split()
                # 0번째가 발화 ID
                utterance_id = parts[0]
                # 1번째가 wav 파일 경로
                wav_path = parts[1]

                # wav 파일을 읽고 특징값 계산
                with wave.open(wav_path) as wav:
                    # 샘플링 주파수 확인
                    if wav.getframerate() != sample_frequency:
                        sys.stderr.write('The expected sampling rate is 16000.\n')
                        exit(1)
                    # wav 파일이 1채널(모노) 데이터인지 확인
                    if wav.getnchannels() != 1:
                        sys.stderr.write('This program supports monaural wav file only.\n')
                        exit(1)

                    # wav 데이터의 샘플 수
                    num_samples = wav.getnframes()

                    # wav 데이터를 읽어들임
                    waveform = wav.readframes(num_samples)

                    # 읽어온 데이터는 바이너리 값(16bit integer)이므로 숫자(정수)로 변환
                    waveform = np.frombuffer(waveform, dtype=np.int16)

                    # Δ와 ΔΔ를 포함한 특징 벡터 계산
                    features_with_delta = feat_extractor.ComputeFeaturesWithDelta(waveform)

                # 특징량의 프레임 수와 차원 수를 가져옴
                (num_frames, num_dims) = np.shape(features_with_delta)

                # 특징값 파일 이름(splitext로 확장자 제거)
                out_file = os.path.join(os.path.abspath(out_dir), utterance_id + '.bin')
                # 출력 경로에 필요한 디렉토리가 없으면 생성
                strbuf = os.path.dirname(out_file)
                if not os.path.exists(strbuf):
                    os.makedirs(strbuf)

                # 데이터를 float32 형식으로 변환
                features_with_delta = features_with_delta.astype(np.float32)

                # 데이터를 파일에 출력
                features_with_delta.tofile(out_file)
                # 발화ID, 특징 파일 경로, 프레임 수, 차원 수를 특징 리스트에 기록
                file_feat.write("%s %s %d %d\n" %
                                (utterance_id, out_file, num_frames, num_dims))


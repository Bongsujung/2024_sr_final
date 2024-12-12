from collections import defaultdict

# wav.scp 파일 경로
wav_scp_path = "./train/wav_all.scp"
output_file_path = "./train/per_500_wav.scp"
max_samples_per_word = 500

# 데이터를 읽어와 단어별로 그룹화
word_dict = defaultdict(list)

with open(wav_scp_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        file_key, file_path = parts
        word = file_key.split('/')[0]  # 단어 추출
        word_dict[word].append(line.strip())

# 최대 500개로 제한하고 부족하면 모두 선택
filtered_data = []
for word, files in word_dict.items():
    filtered_data.extend(files[:max_samples_per_word])

# 출력 파일로 저장
with open(output_file_path, "w") as out_file:
    for line in filtered_data:
        out_file.write(line + "\n")

print(f"총 {len(filtered_data)}개의 발화가 필터링된 데이터셋에 포함되었습니다.")
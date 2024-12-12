# `wav.scp` 파일 경로
wav_scp_path = "./train/wav_all.scp"
output_file_path = "./train/text_phone_large.txt"

# 정의된 단어-발음 매핑 (dic)
dic = {
    "backward": "b ae k w er d",
    "bed": "b eh d",
    "bird": "b er d",
    "cat": "k ae t",
    "dog": "d ao g",
    "down": "d aw n",
    "eight": "ey t",
    "five": "f ay v",
    "follow": "f aa l ow",
    "forward": "f ao r w er d",
    "four": "f ao r",
    "go": "g ow",
    "happy": "hh ae p iy",
    "house": "hh aw s",
    "learn": "l er n",
    "left": "l eh f t",
    "marvin": "m aa r v ih n",
    "nine": "n ay n",
    "no": "n ow",
    "off": "ao f",
    "on": "aa n",
    "one": "hh w ah n",
    "right": "r ay t",
    "seven": "s eh v ah n",
    "sheila": "sh iy l ah",
    "six": "s ih k s",
    "stop": "s t aa p",
    "three": "th r iy",
    "tree": "t r iy",
    "two": "t uw",
    "up": "ah p",
    "visual": "v ih zh ah w ah l",
    "wow": "w aw",
    "yes": "y eh s",
    "zero": "z ih r ow",
}

# `wav.scp` 파일 읽고 단어 추출
text_phone_lines = []
try:
    with open(wav_scp_path, "r") as wav_file:
        for line in wav_file:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            file_key = parts[0]
            word = file_key.split('/')[0]  # 단어 추출
            
            # `dic`에서 단어 확인
            if word in dic:
                text_phone_lines.append(f"{file_key} {dic[word]}")
            else:
                print(f"경고: '{word}' 단어가 정의되지 않았습니다.")

    # `text_phone` 파일 저장
    with open(output_file_path, "w") as text_file:
        for line in text_phone_lines:
            text_file.write(line + "\n")

    print(f"총 {len(text_phone_lines)}개의 발화가 {output_file_path}에 저장되었습니다.")

except FileNotFoundError:
    print(f"파일 {wav_scp_path}을(를) 찾을 수 없습니다.")
except IOError as e:
    print(f"파일 처리 중 오류 발생: {e}")
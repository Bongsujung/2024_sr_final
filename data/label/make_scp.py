import os

def generate_wav_scp(wav_directory, output_file):
    """
    Generate wav.scp with absolute paths for all .wav files in 'wav/' structure.

    :param wav_directory: Root path to the 'wav' folder.
    :param output_file: Path to save the generated wav.scp file.
    """
    with open(output_file, 'w') as scp_file:
        for subfolder in os.listdir(wav_directory):
            subfolder_path = os.path.join(wav_directory, subfolder)
            # Check if the current item is a folder
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".wav"):
                        file_path = os.path.abspath(os.path.join(subfolder_path, file))  # Get absolute path
                        file_id = f"{subfolder}/{os.path.splitext(file)[0]}"
                        scp_file.write(f"{file_id} {file_path}\n")

# Define the root 'wav' directory and output file
wav_dir = "../wav"
output_scp = "./train/wav.scp"

# Generate the wav.scp file
generate_wav_scp(wav_dir, output_scp)
print(f"'wav.scp' 파일이 생성되었습니다: {output_scp}")

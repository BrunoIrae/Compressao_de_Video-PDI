import cv2
import numpy as np

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames

def block_matching(full_search_window, current_block, reference_frame, block_size, start_x, start_y):
    min_mad = float('inf')
    best_match = (0, 0)
    for i in range(-full_search_window, full_search_window + 1):
        for j in range(-full_search_window, full_search_window + 1):
            ref_x = start_x + i
            ref_y = start_y + j
            if 0 <= ref_x <= reference_frame.shape[0] - block_size and 0 <= ref_y <= reference_frame.shape[1] - block_size:
                ref_block = reference_frame[ref_x:ref_x + block_size, ref_y:ref_y + block_size]
                mad = np.mean(np.abs(current_block - ref_block))
                if mad < min_mad:
                    min_mad = mad
                    best_match = (i, j)
    return best_match

def full_search(frames, block_size=16, full_search_window=7):
    motion_vectors = []
    for k in range(1, len(frames)):
        print(f"Processing frame {k}/{len(frames)-1}")
        current_frame = frames[k]
        reference_frame = frames[k - 1]
        frame_motion_vectors = []
        for i in range(0, current_frame.shape[0] - block_size + 1, block_size):
            row_vectors = []
            for j in range(0, current_frame.shape[1] - block_size + 1, block_size):
                current_block = current_frame[i:i + block_size, j:j + block_size]
                best_match = block_matching(full_search_window, current_block, reference_frame, block_size, i, j)
                row_vectors.append(best_match)
            frame_motion_vectors.append(row_vectors)
        motion_vectors.append(frame_motion_vectors)
    return motion_vectors

def apply_motion_vectors(frames, motion_vectors, block_size=16):
    modified_frames = [frames[0]]
    for k in range(1, len(frames)):
        current_frame = frames[k]
        reference_frame = frames[k - 1]
        modified_frame = np.zeros_like(current_frame)
        vectors = motion_vectors[k - 1]
        for i in range(0, current_frame.shape[0] - block_size + 1, block_size):
            for j in range(0, current_frame.shape[1] - block_size + 1, block_size):
                dx, dy = vectors[i // block_size][j // block_size]
                ref_x = i + dx
                ref_y = j + dy
                if 0 <= ref_x <= reference_frame.shape[0] - block_size and 0 <= ref_y <= reference_frame.shape[1] - block_size:
                    modified_frame[i:i + block_size, j:j + block_size] = reference_frame[ref_x:ref_x + block_size, ref_y:ref_y + block_size]
                else:
                    modified_frame[i:i + block_size, j:j + block_size] = current_frame[i:i + block_size, j:j + block_size]
        modified_frames.append(modified_frame)
    return modified_frames

def draw_motion_vectors(frame, motion_vectors, block_size=16):
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for i in range(len(motion_vectors)):
        for j in range(len(motion_vectors[i])):
            dx, dy = motion_vectors[i][j]
            start_x = i * block_size + block_size // 2
            start_y = j * block_size + block_size // 2
            end_x = start_x + dx
            end_y = start_y + dy
            cv2.arrowedLine(color_frame, (start_y, start_x), (end_y, end_x), (0, 0, 255), 2)
    return color_frame

# Adicionando um vídeo
input_video_path = 'C:/Users/McLoving/Documents/Compressão de Video/Data/test.mp4'
frames = load_video(input_video_path)

# Diminuindo o número de frames para visualização rápida
frames = frames[:10]

# Realizando Full Search e gerando vetores de movimento
motion_vectors = full_search(frames, block_size=16, full_search_window=7)

# Aplicando os vetores de movimento para gerar o vídeo modificado
modified_frames = apply_motion_vectors(frames, motion_vectors, block_size=16)

# Exibindo os quadros originais, modificados com setas e modificados sem setas lado a lado em tempo real em uma única janela
for k in range(len(motion_vectors)):
    frame_with_vectors = draw_motion_vectors(frames[k + 1], motion_vectors[k], block_size=16)
    combined_frame = np.hstack((cv2.cvtColor(frames[k + 1], cv2.COLOR_GRAY2BGR),
                                frame_with_vectors,
                                cv2.cvtColor(modified_frames[k + 1], cv2.COLOR_GRAY2BGR)))
    cv2.imshow('Original and Modified Videos', combined_frame)
    if cv2.waitKey(800) & 0xFF == ord('q'):  # Esperar x ms para visualizar o quadro, sair com 'q'
        break

cv2.destroyAllWindows()


def save_video(frames, path, fps=30):
    height, width = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height), isColor=False)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()

# Salvando os vídeos
#save_video(frames, 'original.mp4')
#save_video(modified_frames, 'block_size16.mp4')

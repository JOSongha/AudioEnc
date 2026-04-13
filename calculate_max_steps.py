import math
from typing import List

def calculate_max_steps(
    total_hours: float,
    cutoff_len: int,
    batch_size_per_gpu: int,
    num_gpus: int,
    grad_accum_steps: int,
    tgt_sr: int = 44100,
    hop_length: int = 512,
    proj_strides: List[int] = [2, 2],
    target_epochs: int = 1,
    audio_packing_ratio: float = 0.8
) -> int:
    """
    오디오 총 시간(시간 단위)을 기반으로 분산 학습의 1 에폭당 총 스텝 수를 계산합니다.
    """
    # 1. 1초당 생성되는 오디오 토큰 수 계산
    encoder_fps = tgt_sr / hop_length
    stride_product = math.prod(proj_strides) # 2 * 2 = 4
    tokens_per_second = encoder_fps / stride_product # 약 21.53 토큰/초
    
    # 2. 2만 시간에 대한 전체 오디오 토큰 수 계산
    total_seconds = total_hours * 3600
    total_audio_tokens = total_seconds * tokens_per_second
    
    # 3. 1개 패킹(모델 입력)에 들어갈 수 있는 오디오 토큰 수 
    # (예: cutoff_len이 2048이면 약 1638개의 오디오 토큰이 들어간다고 가정)
    audio_tokens_per_pack = cutoff_len * audio_packing_ratio
    
    # 4. 전체 패킹 샘플(배치 데이터 1단위) 수 추정
    estimated_total_packs = total_audio_tokens / audio_tokens_per_pack
    
    # 5. 글로벌 배치 사이즈 계산
    global_batch_size = num_gpus * batch_size_per_gpu * grad_accum_steps
    
    # 6. 최종 스텝 수 계산 (올림 처리)
    steps_per_epoch = math.ceil(estimated_total_packs / global_batch_size)
    
    return int(steps_per_epoch * target_epochs)

# ==========================================
# 실행 예시 (사용하시는 환경에 맞춰 수정 가능)
# ==========================================
if __name__ == "__main__":
    TOTAL_HOURS = 20000        # 2만 시간
    CUTOFF_LEN = 2048          # 시퀀스 최대 길이
    NUM_GPUS = 8               # GPU 대수
    BATCH_PER_GPU = 4          # GPU 1대당 배치
    GRAD_ACCUM = 4             # 그래디언트 누적

    total_steps = calculate_max_steps(
        total_hours=TOTAL_HOURS,
        cutoff_len=CUTOFF_LEN,
        batch_size_per_gpu=BATCH_PER_GPU,
        num_gpus=NUM_GPUS,
        grad_accum_steps=GRAD_ACCUM
    )
    
    print(f"✅ 오디오 {TOTAL_HOURS}시간 기준 1 에폭당 추정 스텝 수: {total_steps:,} 스텝")
    # (8 GPU, batch 2, accum 4, cutoff 2048 기준 약 14,786 스텝 계산됨)
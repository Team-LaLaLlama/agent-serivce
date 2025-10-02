#!/usr/bin/env python3
"""
독립형 제안서 평가 데이터셋 생성기

특징:
- RFP 파일 불필요: GPT가 직접 RFP 요구사항과 제안서 내용을 생성
- 즉시 실행 가능: API 키만 설정하면 바로 데이터셋 생성
- JSONL 형식: Fine-tuning에 바로 사용 가능한 형식

사용법:
    python standalone_evaluation_generator.py --api-key YOUR_API_KEY --num-samples 5000

    또는 환경변수로:
    export OPENAI_API_KEY=your-api-key-here
    python standalone_evaluation_generator.py --num-samples 5000
"""

import json
import os
import random
import argparse
import glob
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm


# IT 프로젝트 도메인 (5개 분야)
IT_DOMAINS = [
    "IT_시스템구축",
    "데이터_AI",
    "모바일_웹",
    "인프라_보안",
    "IoT_스마트시스템"
]

# 프로젝트 유형 (도메인별)
PROJECT_TYPES = {
    "IT_시스템구축": [
        "전사 ERP 시스템 구축",
        "클라우드 기반 협업 플랫폼",
        "레거시 시스템 현대화",
        "마이크로서비스 아키텍처 전환",
        "통합 인증 시스템 구축"
    ],
    "데이터_AI": [
        "AI 기반 고객 분석 플랫폼",
        "빅데이터 처리 파이프라인",
        "ML 기반 예측 모델링 시스템",
        "실시간 데이터 분석 대시보드",
        "추천 엔진 개발"
    ],
    "모바일_웹": [
        "크로스 플랫폼 모바일 앱",
        "반응형 웹 포털 구축",
        "PWA 기반 서비스",
        "하이브리드 앱 개발",
        "웹 접근성 개선 프로젝트"
    ],
    "인프라_보안": [
        "클라우드 마이그레이션",
        "DevOps 파이프라인 구축",
        "통합 보안 관제 시스템",
        "제로트러스트 보안 아키텍처",
        "DR/BCP 시스템 구축"
    ],
    "IoT_스마트시스템": [
        "IoT 센서 통합 플랫폼",
        "스마트 팩토리 시스템",
        "실시간 모니터링 솔루션",
        "디지털 트윈 구축",
        "엣지 컴퓨팅 플랫폼"
    ]
}

# 평가 유형 및 만점
EVALUATION_TYPES = {
    "전체_평가": {
        "만점": 100,
        "비율": 0.20,
        "설명": "RFP 전체 대비 제안서 전체 평가"
    },
    "기술역량": {
        "만점": 30,
        "비율": 0.15,
        "설명": "기술 아키텍처, 기술 스택, 성능 방안"
    },
    "가격경쟁력": {
        "만점": 25,
        "비율": 0.10,
        "설명": "사업비 적정성, 비용 산정 합리성"
    },
    "수행경험": {
        "만점": 20,
        "비율": 0.10,
        "설명": "유사 프로젝트 실적, 성공 사례"
    },
    "추진전략": {
        "만점": 15,
        "비율": 0.10,
        "설명": "프로젝트 관리, 일정 계획"
    },
    "안정성_리스크관리": {
        "만점": 10,
        "비율": 0.05,
        "설명": "안정화, 모니터링, 리스크 관리"
    },
    "서비스_안정화": {
        "만점": 10,
        "비율": 0.075,
        "설명": "성능 목표, 부하 테스트, 튜닝 전략"
    },
    "장애_대응_복구": {
        "만점": 10,
        "비율": 0.075,
        "설명": "장애 등급, RTO/RPO, 복구 전략"
    },
    "모니터링_관찰성": {
        "만점": 10,
        "비율": 0.075,
        "설명": "모니터링 도구, 지표 수집, 대시보드"
    },
    "리스크_관리": {
        "만점": 10,
        "비율": 0.075,
        "설명": "리스크 식별, 완화 전략, 비상 계획"
    }
}


def generate_score_by_distribution():
    """
    현실적인 점수 분포에 따라 점수 비율 반환 (0.0-1.0)
    """
    rand = random.random()
    if rand < 0.15:
        return random.uniform(0.90, 1.00)  # 90-100%: 15%
    elif rand < 0.40:
        return random.uniform(0.80, 0.89)  # 80-89%: 25%
    elif rand < 0.70:
        return random.uniform(0.70, 0.79)  # 70-79%: 30%
    elif rand < 0.90:
        return random.uniform(0.50, 0.69)  # 50-69%: 20%
    else:
        return random.uniform(0.30, 0.49)  # 30-49%: 10%


def generate_evaluation_sample(client):
    """
    단일 평가 샘플 생성 (instruction-input-output)
    """
    # 랜덤하게 IT 도메인 선택
    domain = random.choice(IT_DOMAINS)
    project_type = random.choice(PROJECT_TYPES[domain])

    # 평가 유형 선택 (비율에 따라)
    rand = random.random()
    cumulative = 0
    eval_type = None
    for et, info in EVALUATION_TYPES.items():
        cumulative += info['비율']
        if rand < cumulative:
            eval_type = et
            break

    if eval_type is None:
        eval_type = "전체_평가"

    max_score = EVALUATION_TYPES[eval_type]['만점']
    eval_description = EVALUATION_TYPES[eval_type]['설명']

    # 점수 생성
    score_ratio = generate_score_by_distribution()
    target_score = int(max_score * score_ratio)

    # GPT에게 RFP 요구사항 + 제안서 내용 + 평가 생성 요청
    prompt = f"""
당신은 IT 프로젝트 제안서 평가 데이터 생성 전문가입니다.

다음 조건으로 평가 샘플을 생성해주세요:

**프로젝트 정보:**
- IT 도메인: {domain}
- 프로젝트 유형: {project_type}

**평가 정보:**
- 평가 항목: {eval_type}
- 평가 설명: {eval_description}
- 목표 점수: {target_score}/{max_score}점

**생성할 내용:**
1. **RFP 요구사항**: {project_type}에 대한 현실적인 요구사항 (2-3문장)
2. **제안서 내용**: 위 RFP에 대한 제안서 내용 (2-3문장)
   - {target_score}/{max_score}점을 받을 만한 수준으로 작성
   - 점수가 낮으면 문제점 포함, 높으면 우수한 내용 포함
3. **평가 코멘트**: RFP 요구사항 대비 제안서의 충족도 평가 (2-3문장)
   - 구체적인 근거 제시
   - {eval_description} 관점에서 평가

**출력 형식 (JSON):**
{{
  "rfp_요구사항": "RFP 요구사항 내용",
  "제안서_내용": "제안서 내용",
  "점수": {target_score},
  "코멘트": "평가 코멘트"
}}

**중요 제약:**
1. 점수는 정확히 {target_score}점이어야 합니다 (만점: {max_score}점)
2. RFP와 제안서는 서로 연관되어야 합니다
3. 평가 코멘트는 RFP 요구사항을 구체적으로 언급해야 합니다
4. IT 도메인({domain})에 맞는 현실적인 내용이어야 합니다
5. 모든 내용은 격식 있는 제안서 문체로 작성하세요
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 IT 프로젝트 제안서 평가 데이터 생성 전문가입니다. 항상 유효한 JSON 형식으로만 응답합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # instruction-input-output 형식으로 변환
        if eval_type == "전체_평가":
            input_text = f"[RFP 요구사항]\n{result['rfp_요구사항']}\n\n[제안서 내용]\n{result['제안서_내용']}"
        else:
            input_text = f"[평가 항목] {eval_type} (만점: {max_score}점)\n[RFP 요구사항]\n{result['rfp_요구사항']}\n\n[제안서 내용]\n{result['제안서_내용']}"

        return {
            "instruction": "심사 기준표에 맞춰 평가 코멘트를 작성하시오.",
            "input": input_text,
            "output": f"[점수] {result['점수']}/{max_score}\n[코멘트] {result['코멘트']}"
        }

    except Exception as e:
        print(f"샘플 생성 실패: {e}")
        return None


def generate_evaluation_dataset(client, num_samples=5000, output_dir="evaluation_training_data"):
    """
    평가 데이터셋을 생성합니다.

    Args:
        client: OpenAI client
        num_samples: 생성할 샘플 수
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"평가 데이터셋 생성 시작: {num_samples}개 샘플")
    print(f"예상 소요 시간: 약 {num_samples * 0.8 / 60:.0f}-{num_samples * 1.2 / 60:.0f}분")
    print(f"{'='*70}\n")

    samples = []
    failed_count = 0

    for i in tqdm(range(num_samples), desc="샘플 생성 중"):
        sample = generate_evaluation_sample(client)

        if sample is not None:
            samples.append(sample)
        else:
            failed_count += 1

        # 중간 저장 (1000개마다)
        if (i + 1) % 1000 == 0 and len(samples) > 0:
            temp_filename = f"{output_dir}/evaluation_dataset_temp_{i+1}.jsonl"
            with open(temp_filename, 'w', encoding='utf-8') as f:
                start_idx = max(0, len(samples) - 1000)
                for s in samples[start_idx:]:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            print(f"\n✅ 중간 저장: {temp_filename} ({len(samples)}개 완료)")

    # 최종 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_dir}/evaluation_dataset_{num_samples}_{timestamp}.jsonl"

    with open(output_filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n{'='*70}")
    print(f"✅ 데이터셋 생성 완료!")
    print(f"  - 성공: {len(samples)}개")
    print(f"  - 실패: {failed_count}개")
    print(f"  - 출력 파일: {output_filename}")
    print(f"{'='*70}\n")

    return samples, output_filename


def analyze_dataset(output_dir="evaluation_training_data"):
    """
    생성된 데이터셋 분석 및 통계 출력
    """
    jsonl_files = sorted(glob.glob(f"{output_dir}/evaluation_dataset_*.jsonl"))

    if not jsonl_files:
        print("생성된 데이터셋 파일이 없습니다.")
        return

    # temp 파일 제외하고 최신 파일 선택
    main_files = [f for f in jsonl_files if 'temp' not in f]
    latest_file = main_files[-1] if main_files else jsonl_files[-1]

    print(f"\n최신 데이터셋 파일: {latest_file}\n")

    # 샘플 로드
    samples = []
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"총 샘플 수: {len(samples)}개\n")

    # 샘플 3개 출력
    print("=== 샘플 예시 ===")
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n[샘플 {i}]")
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input'][:150]}...")
        print(f"Output: {sample['output'][:100]}...")

    # 점수 분포 분석
    print("\n=== 점수 분포 분석 ===")
    scores = []
    for sample in samples:
        output = sample['output']
        if '[점수]' in output:
            score_part = output.split('[점수]')[1].split('[코멘트]')[0].strip()
            score_str = score_part.split('/')[0].strip()
            try:
                scores.append(int(score_str))
            except:
                pass

    if scores:
        score_ranges = {
            "90-100점": 0,
            "80-89점": 0,
            "70-79점": 0,
            "50-69점": 0,
            "30-49점": 0,
            "0-29점": 0
        }

        for score in scores:
            if 90 <= score <= 100:
                score_ranges["90-100점"] += 1
            elif 80 <= score <= 89:
                score_ranges["80-89점"] += 1
            elif 70 <= score <= 79:
                score_ranges["70-79점"] += 1
            elif 50 <= score <= 69:
                score_ranges["50-69점"] += 1
            elif 30 <= score <= 49:
                score_ranges["30-49점"] += 1
            else:
                score_ranges["0-29점"] += 1

        total = len(scores)
        for range_name, count in score_ranges.items():
            percentage = (count / total * 100) if total > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"{range_name:15s}: {bar} {count:4d}개 ({percentage:5.1f}%)")

        print(f"\n평균 점수: {sum(scores)/len(scores):.1f}")
        print(f"최고 점수: {max(scores)}")
        print(f"최저 점수: {min(scores)}")

        # 목표 대비 실제 분포
        print(f"\n=== 목표 대비 실제 분포 ===")
        print(f"90-100점: 목표 15% / 실제 {score_ranges['90-100점']/total*100:.1f}%")
        print(f"80-89점:  목표 25% / 실제 {score_ranges['80-89점']/total*100:.1f}%")
        print(f"70-79점:  목표 30% / 실제 {score_ranges['70-79점']/total*100:.1f}%")
        print(f"50-69점:  목표 20% / 실제 {score_ranges['50-69점']/total*100:.1f}%")
        print(f"30-49점:  목표 10% / 실제 {score_ranges['30-49점']/total*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="독립형 제안서 평가 데이터셋 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 환경변수로 API 키 설정 후 실행
  export OPENAI_API_KEY=your-api-key-here
  python standalone_evaluation_generator.py --num-samples 5000

  # API 키를 직접 지정
  python standalone_evaluation_generator.py --api-key sk-proj-... --num-samples 1000

  # 테스트 (100개)
  python standalone_evaluation_generator.py --num-samples 100

권장 샘플 수:
  - 테스트: 100개 (1-2분)
  - 최소: 1,000개 (10-15분)
  - 권장: 5,000개 (50-70분)
  - 이상적: 10,000개 (100-140분)
        """
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API 키 (환경변수 OPENAI_API_KEY로도 설정 가능)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='생성할 샘플 수 (기본값: 5000)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_training_data',
        help='출력 디렉토리 (기본값: evaluation_training_data)'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='데이터셋 생성 없이 기존 데이터 분석만 수행'
    )

    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')

    if not api_key and not args.analyze_only:
        print("❌ 오류: OpenAI API 키가 필요합니다.")
        print("\n다음 중 한 가지 방법으로 API 키를 설정하세요:")
        print("  1. 환경변수: export OPENAI_API_KEY=your-api-key-here")
        print("  2. 커맨드라인: --api-key your-api-key-here")
        return

    # 분석만 수행
    if args.analyze_only:
        analyze_dataset(args.output_dir)
        return

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    print(f"✅ 환경 설정 완료!")
    print(f"  - IT 도메인: {len(IT_DOMAINS)}개")
    print(f"  - 평가 유형: {len(EVALUATION_TYPES)}개")
    print(f"  - 출력 디렉토리: {args.output_dir}")

    # 데이터셋 생성
    samples, output_file = generate_evaluation_dataset(
        client,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # 생성된 데이터셋 분석
    print("\n데이터셋 분석을 시작합니다...")
    analyze_dataset(args.output_dir)


if __name__ == "__main__":
    main()

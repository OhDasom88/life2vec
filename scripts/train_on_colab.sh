#!/usr/bin/env bash
# life2vec Colab GPU 학습 래퍼
#
# 사용법:
#   ./scripts/train_on_colab.sh auth                          # 최초 1회 OAuth 인증
#   ./scripts/train_on_colab.sh new [GPU]                     # 세션 생성 (기본 A100)
#   ./scripts/train_on_colab.sh sync                          # 코드+데이터 VM 동기화
#   ./scripts/train_on_colab.sh train [EXPERIMENT]            # 학습 실행
#   ./scripts/train_on_colab.sh status                        # 세션 상태
#   ./scripts/train_on_colab.sh download                      # 체크포인트 회수
#   ./scripts/train_on_colab.sh stop                          # 세션 종료
#   ./scripts/train_on_colab.sh run [GPU] [EXPERIMENT]        # new+sync+train 한번에
#
# 예시:
#   ./scripts/train_on_colab.sh run A100 pretrain_agri
#   ./scripts/train_on_colab.sh run A100 finetune_agri_fruiting

set -euo pipefail

SESSION="life2vec"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/content/life2vec"
TARBALL="/tmp/life2vec_code.tar.gz"
DEFAULT_GPU="A100"
DEFAULT_EXPERIMENT="pretrain_agri"

log() { echo "[train_on_colab] $*"; }

resolve_colab() {
  if [[ -n "${COLAB_BIN:-}" && -x "$COLAB_BIN" ]]; then
    echo "$COLAB_BIN"
    return 0
  fi
  if command -v colab >/dev/null 2>&1; then
    command -v colab
    return 0
  fi
  local candidates=(
    "$HOME/miniconda3/bin/colab"
    "$HOME/anaconda3/bin/colab"
    "/opt/conda/bin/colab"
  )
  for bin in "${candidates[@]}"; do
    if [[ -x "$bin" ]]; then
      echo "$bin"
      return 0
    fi
  done
  return 1
}

require_colab() {
  COLAB_BIN="$(resolve_colab)" || {
    echo "colab CLI가 없습니다. (Python 3.12+ 환경에서 pip install google-colab-cli)" >&2
    exit 1
  }
  export COLAB_BIN
}

colab_cmd() {
  require_colab
  "$COLAB_BIN" "$@"
}

session_exists() {
  require_colab
  colab_cmd status -s "$SESSION" >/dev/null 2>&1
}

pack_project() {
  log "프로젝트 패키징 (checkpoints/wandb/logs 제외)..."
  tar czf "$TARBALL" -C "$PROJECT_ROOT" \
    --exclude='checkpoints' \
    --exclude='wandb' \
    --exclude='logs' \
    --exclude='outputs' \
    --exclude='*.egg-info' \
    --exclude='.git' \
    --exclude='__pycache__' \
    src conf data scripts requirements.txt setup.py setup.sh README.md
  log "패키지 크기: $(du -h "$TARBALL" | cut -f1)"
}

cmd_auth_url() {
  require_colab
  local url
  url="$(/home/dasom/miniconda3/bin/python "$PROJECT_ROOT/scripts/colab_oauth.py" url)"
  log "아래 URL을 브라우저에서 열고 인증 코드를 복사하세요:"
  echo ""
  echo "$url"
  echo ""
  log "코드 입력: $0 auth-code <인증코드>"
}

cmd_auth_code() {
  local code="${1:-}"
  require_colab
  [[ -n "$code" ]] || { echo "사용법: $0 auth-code <인증코드>" >&2; exit 1; }
  /home/dasom/miniconda3/bin/python "$PROJECT_ROOT/scripts/colab_oauth.py" finish "$code"
  log "연결 테스트 중..."
  colab_cmd new -s "${SESSION}-auth" --gpu T4
  colab_cmd stop -s "${SESSION}-auth"
  log "인증 및 연결 테스트 완료."
}

cmd_auth() {
  cmd_auth_url
}

cmd_new() {
  local gpu="${1:-$DEFAULT_GPU}"
  require_colab
  if session_exists; then
    log "세션 '$SESSION' 이미 존재합니다."
    colab_cmd status -s "$SESSION"
    return 0
  fi
  log "GPU 세션 생성: $gpu"
  colab_cmd new -s "$SESSION" --gpu "$gpu"
  colab status -s "$SESSION"
}

cmd_sync() {
  require_colab
  session_exists || { log "세션이 없습니다. 먼저: $0 new"; exit 1; }

  pack_project
  log "VM에 업로드 중..."
  colab_cmd upload -s "$SESSION" "$TARBALL" /content/life2vec_code.tar.gz

  log "VM에서 압축 해제..."
  colab_cmd exec -s "$SESSION" <<'PY'
import os, tarfile
os.makedirs("/content/life2vec", exist_ok=True)
with tarfile.open("/content/life2vec_code.tar.gz") as tf:
    tf.extractall("/content/life2vec")
print("extracted to /content/life2vec")
PY

  log "의존성 설치 중 (수 분 소요)..."
  colab_cmd install -s "$SESSION" -r /content/life2vec/requirements.txt
  colab_cmd exec -s "$SESSION" -c "cd /content/life2vec && pip install -e . -q"
  log "동기화 완료."
}

cmd_train() {
  local experiment="${1:-$DEFAULT_EXPERIMENT}"
  require_colab
  session_exists || { log "세션이 없습니다. 먼저: $0 new"; exit 1; }

  log "학습 시작: experiment=$experiment"
  colab_cmd exec -s "$SESSION" -f "$PROJECT_ROOT/scripts/colab_train.py" -- "$experiment"
  log "학습 완료."
}

cmd_download() {
  require_colab
  session_exists || { log "세션이 없습니다."; exit 1; }

  log "원격 체크포인트 목록 확인..."
  colab_cmd ls -s "$SESSION" "$REMOTE_DIR/checkpoints" 2>/dev/null || {
    log "체크포인트 디렉터리가 없습니다."
    return 0
  }

  mkdir -p "$PROJECT_ROOT/checkpoints"
  log "체크포인트 다운로드 중..."
  # 개별 ckpt 파일 다운로드 (디렉터리 재귀 미지원)
  for remote in $(colab_cmd ls -s "$SESSION" "$REMOTE_DIR/checkpoints" 2>/dev/null | grep -E '\.ckpt|\.pth' || true); do
    fname="$(basename "$remote")"
    log "  -> $fname"
    colab_cmd download -s "$SESSION" "$REMOTE_DIR/checkpoints/$fname" "$PROJECT_ROOT/checkpoints/$fname" || true
  done
  log "다운로드 완료: $PROJECT_ROOT/checkpoints/"
}

cmd_status() {
  require_colab
  colab_cmd sessions
  echo "---"
  session_exists && colab_cmd status -s "$SESSION" || log "세션 '$SESSION' 없음"
}

cmd_stop() {
  require_colab
  session_exists && colab_cmd stop -s "$SESSION" || log "종료할 세션 없음"
}

cmd_run() {
  local gpu="${1:-$DEFAULT_GPU}"
  local experiment="${2:-$DEFAULT_EXPERIMENT}"
  cmd_new "$gpu"
  cmd_sync
  cmd_train "$experiment"
  cmd_download
}

usage() {
  sed -n '3,16p' "$0"
}

ACTION="${1:-run}"
shift || true

case "$ACTION" in
  auth)       cmd_auth ;;
  auth-url)   cmd_auth_url ;;
  auth-code)  cmd_auth_code "${1:-}" ;;
  new)      cmd_new "${1:-$DEFAULT_GPU}" ;;
  sync)     cmd_sync ;;
  train)    cmd_train "${1:-$DEFAULT_EXPERIMENT}" ;;
  download) cmd_download ;;
  status)   cmd_status ;;
  stop)     cmd_stop ;;
  run)      cmd_run "${1:-$DEFAULT_GPU}" "${2:-$DEFAULT_EXPERIMENT}" ;;
  help|-h)  usage ;;
  *)        echo "알 수 없는 명령: $ACTION" >&2; usage; exit 1 ;;
esac

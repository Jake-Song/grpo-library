IMAGE="pi-agent"
PROJECT="$(pwd)"
SESSION_DIR="$HOME/.pi-docker-sessions"  # persisted sessions across runs

# -- Load secrets from .env if present --
if [[ -f .env ]]; then
  set -a  # auto-export all subsequent variable assignments
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# -- Build on first use --
if ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Building $IMAGE ... (first run only)"
  docker build -f Dockerfile.pi -t "$IMAGE" .
fi

# -- Session persistence --
mkdir -p "$SESSION_DIR"

# -- Forward all arguments into the container --
docker run -it --rm \
  -v "$PROJECT":/workspace \
  -v "$SESSION_DIR":/root/.pi/agent/sessions \
  -e TERM="${TERM:-xterm-256color}" \
  --workdir /workspace \
  "$IMAGE" \
  "$@"

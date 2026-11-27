#!/usr/bin/env bash
# ==============================================================================
# Server tmux Automation Script:
# 1. Starts the rootless Docker daemon in the left pane via send-keys (loads aliases)
# 2. Splits the window vertically (side-by-side)
# 3. Runs training (scripts/run_server_docker.sh) with forwarded arguments
# 4. Attaches to the tmux session
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="training"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 0. Docker auto-recovery: if the daemon does not respond, clean stale processes and locks
if ! docker info &>/dev/null; then
  echo "🧹 Docker not responding. Cleaning stale processes and orphan locks..."
  pkill -9 -u "$USER" -f "dockerd|rootlesskit|containerd|docker-proxy" 2>/dev/null || true
  USER_DIR="${USER_DOCKER_DIR:-/draft-hdd/$USER/docker-rootless}"
  rm -f "$USER_DIR/.docker/run/docker.pid" "$USER_DIR/.docker/run/docker.sock" 2>/dev/null || true
  rm -f "/draft-hdd/$USER/docker-rootless/.docker/run/docker.pid" "/draft-hdd/$USER/docker-rootless/.docker/run/docker.sock" 2>/dev/null || true
fi

echo "=================================================="
echo "Starting tmux session: $SESSION_NAME"
echo "=================================================="

# 1. Create a detached interactive tmux session (loads ~/.bashrc and aliases)
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"

# 2. Start the Docker daemon in the left pane (.0) resiliently
DAEMON_CMD='if docker info &>/dev/null; then echo "✅ Docker daemon is already active and running."; elif type docker-rootless-daemon-run &>/dev/null; then pkill -9 -u "$USER" -f "dockerd|rootlesskit|containerd|docker-proxy" 2>/dev/null || true; USER_DIR="${USER_DOCKER_DIR:-/draft-hdd/$USER/docker-rootless}"; rm -f "$USER_DIR/.docker/run/docker.pid" "$USER_DIR/.docker/run/docker.sock" 2>/dev/null || true; echo "🚀 Starting Docker Rootless Daemon..."; docker-rootless-daemon-run; else echo "ℹ️ System Docker daemon is ready."; fi'
tmux send-keys -t "$SESSION_NAME.0" "$DAEMON_CMD" C-m

# 3. Split the window horizontally (-h: side-by-side)
tmux split-window -h -t "$SESSION_NAME" -c "$PROJECT_ROOT"

# Brief pause so the shell finishes loading ~/.bashrc before receiving commands
sleep 2

# 4. Assemble arguments and forward environment variables to training
if [ $# -eq 0 ]; then
  TRAIN_ARGS="--map tutorial.txt --num-agents 4"
else
  TRAIN_ARGS="$*"
fi

ENV_PREFIX=""
[ -n "$WORKERS" ] && ENV_PREFIX="WORKERS=$WORKERS $ENV_PREFIX"
[ -n "$CONCURRENT" ] && ENV_PREFIX="CONCURRENT=$CONCURRENT $ENV_PREFIX"
[ -n "$ALGO" ] && ENV_PREFIX="ALGO=$ALGO $ENV_PREFIX"
[ -n "$ALGOS" ] && ENV_PREFIX="ALGOS=\"$ALGOS\" $ENV_PREFIX"
[ -n "$FAIL_FAST" ] && ENV_PREFIX="FAIL_FAST=$FAIL_FAST $ENV_PREFIX"
[ -n "$EVAL_EPISODES" ] && ENV_PREFIX="EVAL_EPISODES=$EVAL_EPISODES $ENV_PREFIX"

# Send the command to the right pane (.1), dynamically waiting for Docker to respond
TRAIN_CMD="echo '⏳ Waiting for Docker daemon to respond...'; for i in {1..30}; do if docker info &>/dev/null; then echo '✅ Docker connected successfully!'; break; fi; sleep 1; done; ${ENV_PREFIX}bash scripts/run_server_docker.sh $TRAIN_ARGS; echo ''; echo '🏁 Process finished with exit code: '\$?; echo 'Press Enter to close this pane.'; read"
tmux send-keys -t "$SESSION_NAME.1" "$TRAIN_CMD" C-m

# 5. Attach directly to the session for live monitoring
tmux attach-session -t "$SESSION_NAME"

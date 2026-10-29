#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
port="${FDM_REMOTE_PORT:-8502}"
if ! .venv/bin/python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$port"; then
  echo "Port $port is busy. Choose a free port, e.g. FDM_REMOTE_PORT=$((port + 1)) bash tools/serve_remote.sh"
  exit 1
fi
mkdir -p .tools logs
if command -v cloudflared >/dev/null 2>&1; then
  tunnel_bin="$(command -v cloudflared)"
else
  tunnel_bin="$PWD/.tools/cloudflared"
  if [[ ! -x "$tunnel_bin" ]]; then
    case "$(uname -m)" in
      x86_64) arch=amd64 ;;
      aarch64) arch=arm64 ;;
      *) echo "Install cloudflared for this architecture first."; exit 1 ;;
    esac
    echo "Downloading cloudflared once into this project (.tools/)."
    curl --fail --location --retry 2 --connect-timeout 15 --max-time 300 \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch" \
      --output "$tunnel_bin.download"
    chmod +x "$tunnel_bin.download"
    mv "$tunnel_bin.download" "$tunnel_bin"
  fi
fi
app_pid=""
tunnel_pid=""
cleanup() {
  [[ -z "$tunnel_pid" ]] || kill "$tunnel_pid" 2>/dev/null || true
  [[ -z "$app_pid" ]] || kill "$app_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
echo "Starting FDM app (TensorFlow uses the GPU when available)."
.venv/bin/python -u src/remote_camera_server.py --port "$port" \
  > logs/remote_app.log 2>&1 &
app_pid=$!
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$app_pid" 2>/dev/null; then cat logs/remote_app.log; exit 1; fi
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then echo "App did not become ready. See logs/remote_app.log"; exit 1; fi
echo "Share only the HTTPS trycloudflare.com link printed below with your friend."
echo "Anyone with that link can access the app. Keep this terminal and laptop running."
echo "Press Ctrl+C to shut down both the app and the public link."
"$tunnel_bin" tunnel --url "http://127.0.0.1:$port" --no-autoupdate &
tunnel_pid=$!
wait -n "$app_pid" "$tunnel_pid"

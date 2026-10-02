#!/usr/bin/env bash
# Keeps the bot running: restarts it a minute after it exits for any reason.
# Run it inside tmux so it survives the Session Manager window closing.
#
#   scripts/run_bot.sh test              # testing account
#   scripts/run_bot.sh comp              # competition account
#   scripts/run_bot.sh test --dry-run    # extra arguments go to python -m bot.live
#
# Credentials are read from ~/.roostoo_<account>.env, which must export
# ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY. To deploy new code without stopping
# this loop:  git pull && pkill -f "bot.live --account comp"
set -u

account="${1:?usage: scripts/run_bot.sh test|comp [bot.live options]}"
shift
env_file="$HOME/.roostoo_${account}.env"
cd "$(dirname "$0")/.." || exit 1
mkdir -p "runs/$account"

trap 'echo "run_bot: stopped"; exit 0' INT

while true; do
    if [ ! -f "$env_file" ]; then
        echo "run_bot: $env_file not found" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    . "$env_file"
    python3 -m bot.live --account "$account" "$@"
    code=$?
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') bot exited with code $code, restarting in 60s" \
        | tee -a "runs/$account/restarts.log"
    sleep 60
done

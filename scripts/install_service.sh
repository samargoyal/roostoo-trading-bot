#!/usr/bin/env bash
# Runs the bot as a systemd service, so it comes back after the server reboots (a tmux session
# does not). Stop any copy started by hand first (in its tmux window: Ctrl-c), then:
#
#   sudo scripts/install_service.sh comp
#
# The service runs scripts/run_bot.sh, which reads ~/.roostoo_<account>.env and restarts the
# bot a minute after it exits. Deploy new code as before: git pull && pkill -f "bot.live --account comp"
set -eu

account="${1:?usage: sudo scripts/install_service.sh test|comp}"
case "$account" in
    *[!A-Za-z0-9_-]*) echo "install_service: bad account name: $account" >&2; exit 1 ;;
esac
dir="$(cd "$(dirname "$0")/.." && pwd)"
user="${SUDO_USER:-$(id -un)}"
unit="roostoo-bot-$account"

if pgrep -f "bot.live --account $account" >/dev/null; then
    echo "install_service: a $account bot is already running; stop it first (in tmux: Ctrl-c)" >&2
    exit 1
fi

cat > "/etc/systemd/system/$unit.service" <<UNIT
[Unit]
Description=Roostoo trading bot ($account account)
After=network-online.target
Wants=network-online.target

[Service]
User=$user
WorkingDirectory=$dir
ExecStart=/bin/bash $dir/scripts/run_bot.sh $account
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now "$unit"
echo "started $unit: systemctl status $unit; tail -f $dir/runs/$account/logs/bot.log"

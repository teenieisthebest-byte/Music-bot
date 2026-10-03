#!/usr/bin/env bash
# One-shot setup for an Ubuntu server (e.g. Oracle Cloud free VM).
# Installs everything and runs the bot as a service that starts on boot
# and restarts itself if it crashes.
set -e
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libopus0 libffi-dev

python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt

read -rsp "Discord bot token: " TOKEN; echo
cat > .env <<ENVFILE
DISCORD_TOKEN=$TOKEN
# Optional - uncomment and fill in:
# YTM_EMOJI=<:ytmusic:1234567890>
# SPOTIPY_CLIENT_ID=
# SPOTIPY_CLIENT_SECRET=
ENVFILE
chmod 600 .env

sudo tee /etc/systemd/system/musicbot.service >/dev/null <<UNIT
[Unit]
Description=Discord music bot
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$PWD
EnvironmentFile=$PWD/.env
ExecStart=$PWD/venv/bin/python bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now musicbot
echo "Done! The bot is running. Check logs with: journalctl -u musicbot -f"

#!/bin/bash
# Makes trashbot start by itself at power-on. Run once:   ./install_service.sh
# (asks for your password because it changes system settings)
set -e
cd "$(dirname "$0")"

echo "1/4 Installing the trashbot service..."
sudo cp trashbot.service /etc/systemd/system/trashbot.service
sudo systemctl daemon-reload
sudo systemctl enable trashbot

echo "2/4 Keeping logs across reboots (so crashes can be investigated)..."
sudo mkdir -p /var/log/journal
sudo systemctl restart systemd-journald

echo "3/4 Hardware watchdog: if the whole Pi ever freezes, it reboots itself..."
sudo mkdir -p /etc/systemd/system.conf.d
printf "[Manager]\nRuntimeWatchdogSec=15\n" | sudo tee /etc/systemd/system.conf.d/watchdog.conf >/dev/null
sudo systemctl daemon-reexec

echo "4/4 Starting trashbot now (keep the tray EMPTY - it calibrates first)..."
sudo systemctl restart trashbot
sleep 2
systemctl --no-pager status trashbot | head -5
echo
echo "Done. It now starts at every power-on."
echo "  live logs:      journalctl -u trashbot -f"
echo "  stop / start:   sudo systemctl stop trashbot    sudo systemctl start trashbot"
echo "  turn off auto:  sudo systemctl disable trashbot"

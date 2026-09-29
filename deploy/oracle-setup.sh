#!/usr/bin/env bash
# Installs or updates Gel Band Analyzer on an Oracle Cloud Ubuntu server.
# Run it again at any time to update to the latest code on main.
# Optional: DOMAIN=gels.example.org bash oracle-setup.sh to use your own domain.
set -euo pipefail

REPO=https://github.com/JPmarsxxxi/gel-blot-analyzer.git
SRC=$HOME/gel-blot-analyzer
DATA=/srv/gel-data

if ! command -v docker >/dev/null || ! command -v git >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker.io git
fi

# Oracle's Ubuntu images reject inbound traffic in the iptables INPUT chain,
# but Docker's published ports are forwarded to the container and never pass
# through INPUT, so only the cloud firewall (security list) needs opening.

if [ -d "$SRC/.git" ]; then
  git -C "$SRC" pull --ff-only
else
  git clone --depth 1 "$REPO" "$SRC"
fi

# sslip.io resolves <ip>.sslip.io to <ip>, which gives a free hostname that
# Caddy can get an HTTPS certificate for.
if [ -z "${DOMAIN:-}" ]; then
  DOMAIN="$(curl -fsS https://api.ipify.org).sslip.io"
fi

# The container runs as uid 1000 and stores projects in /data.
sudo mkdir -p "$DATA"
sudo chown 1000:1000 "$DATA"

sudo docker build -t gel-analyzer "$SRC"
sudo docker network inspect gel >/dev/null 2>&1 || sudo docker network create gel

sudo docker rm -f gel-app >/dev/null 2>&1 || true
sudo docker run -d --name gel-app --network gel --restart unless-stopped \
  -v "$DATA":/data gel-analyzer

sudo docker rm -f gel-caddy >/dev/null 2>&1 || true
sudo docker run -d --name gel-caddy --network gel --restart unless-stopped \
  -p 80:80 -p 443:443 -v gel-caddy-data:/data \
  caddy:2 caddy reverse-proxy --from "$DOMAIN" --to gel-app:7860

echo
echo "Done. In a minute or two the app is live at: https://$DOMAIN"

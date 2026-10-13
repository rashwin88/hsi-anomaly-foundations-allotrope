#!/usr/bin/env bash
# One-time host setup for the Allotrope EC2 box. Run as root (via SSM).
# Installs Docker Engine + compose plugin, AWS CLI v2, git, and creates a
# read-only GitHub deploy key whose public half must be added to the repo
# (Settings → Deploy keys) before clone.sh can run. Idempotent.
set -euo pipefail

apt-get update -q
apt-get install -y -q ca-certificates curl git unzip jq

if ! command -v docker >/dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -q
    apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! command -v aws >/dev/null; then
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q -o /tmp/awscliv2.zip -d /tmp && /tmp/aws/install
fi

key=/root/.ssh/allotrope_deploy
if [ ! -f "$key" ]; then
    install -m 0700 -d /root/.ssh
    ssh-keygen -q -t ed25519 -N "" -C "allotrope-prod-deploy" -f "$key"
    ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
    printf 'Host github.com\n  IdentityFile %s\n  IdentitiesOnly yes\n' "$key" >> /root/.ssh/config
fi

echo "== docker $(docker --version | cut -d' ' -f3) | $(aws --version 2>&1 | cut -d' ' -f1)"
echo "== add this READ-ONLY deploy key to the GitHub repo:"
cat "$key.pub"

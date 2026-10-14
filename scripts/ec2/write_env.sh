#!/usr/bin/env bash
# Write docker/.env and backend/.env.inventory from SSM Parameter Store
# (/allotrope/prod/*, SecureString), using the instance role. Run as root
# from the repo root on the box. Secrets never live in git or on a laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."

get() { aws ssm get-parameter --region ap-south-1 --with-decryption \
            --name "/allotrope/prod/$1" --query Parameter.Value --output text; }

umask 077
cat > docker/.env <<ENV
POSTGRES_PASSWORD=$(get postgres_password)
ADMIN_USERNAME=$(get admin_username)
ADMIN_EMAIL=$(get admin_email)
ADMIN_PASSWORD=$(get admin_password)
JWT_SECRET=$(get jwt_secret)
INVENTORY_UI_KEY=$(get inventory_ui_key)
SITE_ADDRESS=$(get site_address)
ENV

# No AWS keys: the inventory container uses the instance role (IMDSv2,
# hop limit 2 so containers can reach it).
cat > backend/.env.inventory <<ENV
INVENTORY_DB_HOST=aws-0-ap-south-1.pooler.supabase.com
INVENTORY_DB_PORT=5432
INVENTORY_DB_USER=postgres.eifymxkorelthicwnyxc
INVENTORY_DB_NAME=postgres
INVENTORY_DB_PASSWORD=$(get inventory_db_password)
INVENTORY_API_KEYS=$(get inventory_api_keys)
ENV
echo "wrote docker/.env and backend/.env.inventory (mode 600)"

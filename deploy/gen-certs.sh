#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/certs"
pushd "$DIR/certs" >/dev/null

openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ca.key -out ca.pem -subj "/CN=dev-ca" >/dev/null 2>&1
openssl req -new -nodes -newkey rsa:2048 -keyout server.key -out server.csr -subj "/CN=dev-server" >/dev/null 2>&1
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out server.crt -days 365 >/dev/null 2>&1
echo "Wrote certs to $DIR/certs"
popd >/dev/null



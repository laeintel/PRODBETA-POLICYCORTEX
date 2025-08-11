#!/bin/sh
echo "Starting mock PolicyCortex Core server (compilation issues being resolved)..."
while true; do
  echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 195\r\n\r\n{\"status\":\"healthy\",\"version\":\"2.0.0-mock\",\"service\":\"policycortex-core\",\"note\":\"Mock server active while compilation issues are resolved\",\"timestamp\":\"$(date -Iseconds)\"}" | nc -l -p 8080 -q 1
done
#!/usr/bin/env bash
# down.sh -- stop the server started by ./up.sh and show that the cards are free.
set -euo pipefail
NAME=radiance
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "stopping $NAME (graceful, up to 30 s) …"
  docker stop -t 30 "$NAME" >/dev/null
  echo "stopped; its log stays readable with: docker logs $NAME"
else
  echo "$NAME is not running."
fi
# VRAM per card, from sysfs. A card in deep sleep cannot be read without waking it: that is idle.
for d in /sys/class/drm/card*/device; do
  [ -r "$d/mem_info_vram_used" ] || continue
  [ "$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)" -ge $((8 << 30)) ] || continue
  if [ "$(cat "$d/power/runtime_status" 2>/dev/null)" != active ]; then
    printf '  %s  asleep (idle)\n' "$(basename "$(readlink -f "$d")")"; continue
  fi
  printf '  %s  VRAM used: %5d of %d MiB\n' "$(basename "$(readlink -f "$d")")" \
    $(( $(cat "$d/mem_info_vram_used") >> 20 )) $(( $(cat "$d/mem_info_vram_total") >> 20 ))
done

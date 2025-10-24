#!/usr/bin/env bash
set -euo pipefail

# ---- Settings (tweak as you like) ----
ISO_URL="https://releases.ubuntu.com/24.04/ubuntu-24.04.1-live-server-arm64.iso"
ISO_FILE="ubuntu-24.04.1-live-server-arm64.iso"
DISK="ubuntu-arm64.qcow2"
DISK_SIZE="40G"
RAM="6G"
CPUS="4"
NAME="ubuntu-arm64"

# Locate QEMU's AArch64 UEFI firmware (Homebrew install)
BREW_PREFIX="$(brew --prefix)"
FIRMWARE="${BREW_PREFIX}/share/qemu/edk2-aarch64-code.fd"

# ---- Fetch installer ISO if missing ----
if [ ! -f "$ISO_FILE" ]; then
  echo "Downloading $ISO_FILE ..."
  curl -L -o "$ISO_FILE" "$ISO_URL"
fi

# ---- Create disk if missing ----
if [ ! -f "$DISK" ]; then
  echo "Creating $DISK ($DISK_SIZE) ..."
  qemu-img create -f qcow2 "$DISK" "$DISK_SIZE"
fi

# ---- Sanity checks ----
if [ ! -f "$FIRMWARE" ]; then
  echo "Cannot find firmware: $FIRMWARE"
  echo "On Homebrew, it’s typically at /opt/homebrew/share/qemu/edk2-aarch64-code.fd"
  echo "Install/upgrade qemu via Homebrew if needed: brew install qemu"
  exit 1
fi

# ---- Run QEMU with GUI + keyboard/mouse ----
exec qemu-system-aarch64 \
  -name "$NAME" \
  -machine virt,accel=hvf,highmem=on \
  -cpu host -smp "$CPUS" -m "$RAM" \
  -bios "$FIRMWARE" \
  -display cocoa \
  -device virtio-gpu-pci \
  -device virtio-keyboard-pci \
  -device virtio-tablet-pci \
  -netdev user,id=n0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=n0 \
  -drive if=virtio,file="$DISK",format=qcow2 \
  -cdrom "$ISO_FILE" -boot d

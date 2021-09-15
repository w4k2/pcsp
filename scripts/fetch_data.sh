#!/usr/bin/env bash

# Download gdown
# `python -m pip install gdown`

# cse
# https://drive.google.com/file/d/1YPIN14rl1ITdkx110k4Om6RzuXFaDkHf/view?usp=sharing
# deric
# https://drive.google.com/file/d/1QD4w-KwcFDGTC1bDAo3vr6uYJoiS0-wX/view?usp=sharing
# uea_ucr
# https://drive.google.com/file/d/1OKarlJrv-_fidMu7s4B9-FeLs1lz2kqf/view?usp=sharing

SCRIPT_DIR=$(dirname "$0")
REPO_DIR=$(realpath "${SCRIPT_DIR}/..")
DATA_DIR="${REPO_DIR}/data"
TMPFILE=$(mktemp "/tmp/in-waves.XXXXXX")

mkdir -p ${DATA_DIR}

gdown "https://drive.google.com/uc?export=download&id=1YPIN14rl1ITdkx110k4Om6RzuXFaDkHf" -O ${TMPFILE}
unzip ${TMPFILE} -d "${DATA_DIR}"

gdown "https://drive.google.com/uc?export=download&id=1QD4w-KwcFDGTC1bDAo3vr6uYJoiS0-wX" -O ${TMPFILE}
unzip ${TMPFILE} -d "${DATA_DIR}"

rm -v "$TMPFILE"

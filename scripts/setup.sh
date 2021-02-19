#!/bin/bash

# install dependencies
pip install --user -r requirements.txt

# setup CUB-200-2011 dataset
DATA_SETUP=false
while :; do
  case $1 in
    -d|--data_setup)
    DATA_SETUP=true
    ;;
    *) break
  esac
  shift
done

if [[ "$DATA_SETUP" = true ]]; then
  echo 'Setting up CUB-200-2011...'
  python setup.py
  echo 'Setup for CUB-200-2011 is completed.'
fi


#!/bin/sh

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -t|--tag)
      TAG="$2"
      shift # past argument
      ;;
      -d|--train_dir)
      TRAIN_DIR="$2"
      shift # past argument
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

python train.py \
    --config_path=${CONFIG_PATH} \
    --tag=${TAG} \
    --train_dir=${TRAIN_DIR}

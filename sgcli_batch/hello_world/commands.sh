#!/bin/bash
echo ">>> SGC Hello World — Starting"
echo ">>> Node Rank: $NODE_RANK"
echo ">>> World Size: $WORLD_SIZE"

python train.py

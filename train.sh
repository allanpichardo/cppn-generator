#!/usr/bin/env bash

lightweight_gan \
    --data ./path/to/images \
    --name {name of run} \
    --batch-size 16 \
    --gradient-accumulate-every 4 \
    --num-train-steps 200000 \
    --aug-prob 0.25 \
    --aug-types [translation,cutout,color] \
    --image-size 128 \
    --amp \
    --dual-contrast-loss \
    --transparent \
    --use-aim \
    --aim_repo ~/cppn-logs
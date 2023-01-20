#!/usr/bin/env bash

lightweight_gan \
    --data /home/allan/portrait-generator/collages \
    --name cppn-test-2 \
    --batch-size 24 \
    --gradient-accumulate-every 4 \
    --num-train-steps 200000 \
    --aug-prob 0.25 \
    --aug-types [translation,cutout,color] \
    --image-size 64 \
    --dual-contrast-loss \
    --use-aim \
    --aim_repo /home/allan/cppn-logs
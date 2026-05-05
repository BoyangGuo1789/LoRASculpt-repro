#!/bin/bash

export IDS_ENABLE=${IDS_ENABLE:-True}
export IDS_GRAD_LAMBDA=${IDS_GRAD_LAMBDA:-1.0}
export IDS_RET_LAMBDA=${IDS_RET_LAMBDA:-0.5}
export IDS_GRAD_EMA_BETA=${IDS_GRAD_EMA_BETA:-0.9}
export IDS_APPLY_RETENTION_TO=${IDS_APPLY_RETENTION_TO:-qkv}
export IDS_SCORE_EPS=${IDS_SCORE_EPS:-1e-8}
export RUN_NAME=${RUN_NAME:-llava-v1.5-7b-lorasculpt-dis-iconqa-r32}

#!/usr/bin/env bash

set -a && . .env && set +a

python jobby.py $@

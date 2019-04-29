#!/usr/bin/env bash

set -a && . .env && set +a

if [[ -z "${JOBBY_JOB_CONCURRENCY}" ]]
then
    echo "[ERROR] JOBBY_JOB_CONCURRENCY is not set"
    exit 1
fi

RANDOM_STR=$(hexdump -n 2 -ve '/1 "%02X"' -e '/2 "\n"' /dev/urandom)
DATE_STR=$(date +%s)
WORKER_ID=worker.${DATE_STR}.${RANDOM_STR}@%h

celery worker -A jobby \
    -l info -Ofair \
    -n ${WORKER_ID} \
    -c ${JOBBY_JOB_CONCURRENCY}

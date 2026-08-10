#!/bin/sh

credential_file=${PDCS_GITEE_CREDENTIAL_FILE:?missing PDCS_GITEE_CREDENTIAL_FILE}

case "$1" in
    *Username*) key=username ;;
    *Password*) key=passwd ;;
    *) exit 1 ;;
esac

awk -F: -v requested_key="$key" '
    $1 == requested_key {
        value = substr($0, index($0, ":") + 1)
        sub(/^[[:space:]]+/, "", value)
        sub(/[[:space:]]+$/, "", value)
        print value
        exit
    }
' "$credential_file"

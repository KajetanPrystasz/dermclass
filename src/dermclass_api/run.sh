#!/usr/bin/env bash
#TODO: Fix logging
export IS_DEBUG=${DEBUG:-true}
exec gunicorn -b :${PORT:-5000} --access-logfile - --error-logfile - run:application

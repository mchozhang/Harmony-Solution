#!/bin/bash
# uwsgi daemon script
# copy this file to /etc/init

start() {
    uwsgi /var/www/harmony-solution/uwsgi/uwsgi.ini
}

case "$1" in
    start)
        start;;
    *)
        echo "Usage: $0 {start}"
esac

exit 0


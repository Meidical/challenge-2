#!/bin/bash
docker rmi $(docker images -f "dangling=true" -q)

docker-compose down
docker-compose up -d
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --user-data-dir="/tmp/chrome-dev-session" --disable-web-security "index.html"
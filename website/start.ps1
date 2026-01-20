# Stop running containers if present
docker-compose down

# Start containers
docker-compose up -d

# Open Chrome at localhost:3000 with custom flags (Windows)
$site = Join-Path -Path $pwd -ChildPath 'index.html'
Start-Process -FilePath "chrome.exe" -ArgumentList "--user-data-dir=C:\Chrome dev session", "--disable-web-security", $site
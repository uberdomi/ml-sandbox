# 🧪 ML Sandbox

Ein Docker-basiertes Machine Learning Playground-Setup.

## 🚀 Start

1. Repository klonen:
   ```bash
   git clone https://github.com/<dein-username>/ml-sandbox.git
   cd ml-sandbox
   ```

2. Container bauen & starten:
   ```bash
   docker-compose up --build
   ```

3. JupyterLab im Browser öffnen:
   [http://localhost:8888](http://localhost:8888)  
   → Token steht im Terminal.

## 📂 Struktur
- `notebooks/` → deine Experimente
- `requirements.txt` → Python-Pakete
- `Dockerfile` → Umgebung
- `docker-compose.yml` → einfacher Start

## 🛠 Anpassungen
- Weitere Pakete in `requirements.txt` eintragen.
- Container neu bauen:  
  ```bash
  docker-compose build
  ```

LOG_FILE="log/log-$(date +%F_%H-%M-%S).txt"

echo "🧼 Corrigindo permissões (caso necessárias)..."
chmod +x start.sh

echo "🚀 Iniciando Docker Compose e salvando logs em $LOG_FILE ..."
docker compose up --build > "$LOG_FILE" 2>&1 &

sleep 5

echo "🌐 Tentando abrir no navegador: http://localhost:8501"
( which wslview > /dev/null && wslview http://localhost:8501 ) || powershell.exe start http://localhost:8501

echo "✅ Tudo pronto. Logs estão sendo gravados em: $LOG_FILE"

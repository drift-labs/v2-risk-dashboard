workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = 1200
keepalive = 65
max_requests = 1000
max_requests_jitter = 50
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Restart workers that die unexpectedly
worker_exit_on_restart = True
worker_restart_delay = 20
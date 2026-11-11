#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
build_dir=$(cd "$repo_dir/.." && pwd)

cd "$build_dir"
if [[ "${1:-}" != "--skip-build" ]]; then
    minikube image build --all -f rust-port/Dockerfile.worker -t llama2-worker:dev .
fi
kubectl apply -f "$repo_dir/k8s/worker.yaml"
kubectl apply -f "$repo_dir/k8s/prometheus.yaml"
kubectl apply -f "$repo_dir/k8s/grafana.yaml"
if [[ "${1:-}" != "--skip-build" ]]; then
    kubectl rollout restart statefulset/llama2-worker
fi
kubectl rollout status statefulset/llama2-worker --timeout=180s
kubectl rollout status deployment/llama2-prometheus --timeout=180s
kubectl rollout status deployment/llama2-grafana --timeout=180s

kubectl port-forward pod/llama2-worker-0 18080:8080 &
worker_0=$!
kubectl port-forward pod/llama2-worker-1 18082:8080 &
worker_1=$!
kubectl port-forward deployment/llama2-prometheus 19090:9090 &
prometheus=$!
kubectl port-forward deployment/llama2-grafana 13000:3000 &
grafana=$!

cleanup() {
    kill "$worker_0" "$worker_1" "$prometheus" "$grafana" 2>/dev/null || true
    wait "$worker_0" "$worker_1" "$prometheus" "$grafana" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1
if ! kill -0 "$worker_0" 2>/dev/null || ! kill -0 "$worker_1" 2>/dev/null || ! kill -0 "$prometheus" 2>/dev/null || ! kill -0 "$grafana" 2>/dev/null; then
    echo "A port-forward failed. Check whether ports 18080, 18082, 19090, and 13000 are already in use." >&2
    exit 1
fi

cat > "$repo_dir/router/workers.minikube.yaml" <<'YAML'
listen: 127.0.0.1:8081
health_interval_ms: 2000
workers:
  - http://127.0.0.1:18080
  - http://127.0.0.1:18082
YAML

printf 'Workers forwarded to http://127.0.0.1:18080 and http://127.0.0.1:18082\n'
printf 'Prometheus: http://127.0.0.1:19090/targets\n'
printf 'Grafana dashboard: http://127.0.0.1:13000/d/llama2-workers\n'
printf 'Host router config: %s/router/workers.minikube.yaml\n' "$repo_dir"
printf 'Run: cd %s && cargo run --release -p llama2-router -- router/workers.minikube.yaml\n' "$repo_dir"
printf 'Keep this terminal open for the port-forwards. Press Ctrl-C to stop them.\n'
wait

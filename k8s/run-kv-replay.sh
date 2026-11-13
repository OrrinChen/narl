#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
root=$(pwd -P)
mkdir -p runs
out=$(mktemp -d "$root/runs/k8s-kv-replay.XXXXXX")
cluster="narl-$(printf '%s' "${out##*.}" | tr '[:upper:]' '[:lower:]')"
image="narl-kv-replay:$cluster"
kubeconfig=$(mktemp)
kube=(kubectl --kubeconfig "$kubeconfig" --context "kind-$cluster")
cleanup() {
    kind delete cluster --name "$cluster" --kubeconfig "$kubeconfig" || true
    rm -f -- "$kubeconfig"
}
trap cleanup EXIT
printf 'Job artifacts: %s\n' "$out"

source_commit=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain) ]]; then
    source_commit="$source_commit-dirty"
fi
docker build --build-arg "NARL_SOURCE_COMMIT=$source_commit" \
    -f k8s/Dockerfile -t "$image" .
docker image inspect --format '{{.Id}}' "$image" > "$out/image-id.txt"
docker run --rm --network none "$image" python -m pip freeze > "$out/requirements-resolved.txt"

kind create cluster --name "$cluster" --kubeconfig "$kubeconfig" --wait 120s --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  apiServerAddress: "127.0.0.1"
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: "$out"
        containerPath: /narl-results
EOF
kind load docker-image "$image" --name "$cluster"
"${kube[@]}" version -o json > "$out/kubernetes-version.json"
"${kube[@]}" set image --local -f k8s/kv-replay-job.yaml "kv-replay=$image" -o yaml \
    > "$out/manifest.yaml"
"${kube[@]}" create -f "$out/manifest.yaml"

status=0
"${kube[@]}" wait --for=condition=complete --timeout=200s job/narl-kv-replay || status=$?
"${kube[@]}" get job narl-kv-replay -o json > "$out/job.json"
"${kube[@]}" get pods -l job-name=narl-kv-replay -o json > "$out/pods.json"
"${kube[@]}" logs -l job-name=narl-kv-replay --prefix --tail=-1 > "$out/job.log" 2>&1
if [[ $status -ne 0 ]]; then
    exit "$status"
fi

pod_uid=$("${kube[@]}" get pods -l job-name=narl-kv-replay \
    --field-selector=status.phase=Succeeded -o jsonpath='{.items[0].metadata.uid}')
test -s "$out/job.log" && test -s "$out/$pod_uid/summary.json" && test -s "$out/$pod_uid/timings.csv"
printf 'Completed. Benchmark result: %s/%s/summary.json\n' "$out" "$pod_uid"

#[allow(dead_code)] // Nodes also keep their word values for inspection.
mod tree;

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use tree::PrefixTree;

const BALANCE_ABS_THRESHOLD: usize = 1;
const BALANCE_REL_THRESHOLD: f32 = 1.5;

#[derive(Deserialize)]
struct Config {
    listen: String,
    health_interval_ms: u64,
    workers: Vec<String>,
}

#[derive(Clone, Serialize)]
struct Health {
    healthy: bool,
    checked_at_ms: Option<u128>,
    error: Option<String>,
}

struct Worker {
    url: String,
    health: Mutex<Health>,
    pending_requests: AtomicUsize,
}

struct PendingRequest<'a>(&'a AtomicUsize);

impl<'a> PendingRequest<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for PendingRequest<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Cluster {
    workers: Vec<Worker>,
    next: AtomicUsize,
    tree: Mutex<PrefixTree>,
}

impl Cluster {
    fn new(urls: Vec<String>) -> Self {
        let worker_count = urls.len();
        Self {
            workers: urls
                .into_iter()
                .map(|url| Worker {
                    url: url.trim_end_matches('/').to_owned(),
                    health: Mutex::new(Health {
                        healthy: false,
                        checked_at_ms: None,
                        error: None,
                    }),
                    pending_requests: AtomicUsize::new(0),
                })
                .collect(),
            next: AtomicUsize::new(0),
            tree: Mutex::new(PrefixTree::new(worker_count)),
        }
    }

    fn healthy_candidates(&self, candidates: &[usize]) -> Vec<usize> {
        candidates
            .iter()
            .copied()
            .filter(|&index| self.workers[index].health.lock().unwrap().healthy)
            .collect()
    }

    fn select_worker(&self, prompt: &str) -> Option<(usize, &Worker, PendingRequest<'_>)> {
        let tree = self.tree.lock().unwrap();
        let healthy = self.healthy_candidates(tree.find("").workers);
        if healthy.is_empty() {
            return None;
        }

        let loads: Vec<_> = healthy
            .iter()
            .map(|&index| self.workers[index].pending_requests.load(Ordering::Relaxed))
            .collect();
        let min = *loads.iter().min().unwrap();
        let max = *loads.iter().max().unwrap();
        let imbalanced = max - min > BALANCE_ABS_THRESHOLD
            && (max as f32) > (min as f32) * BALANCE_REL_THRESHOLD;

        let candidates = if imbalanced {
            healthy
                .iter()
                .zip(&loads)
                .filter_map(|(&index, &load)| (load == min).then_some(index))
                .collect()
        } else {
            let matches = self.healthy_candidates(tree.find(prompt).workers);
            if matches.is_empty() {
                healthy
            } else {
                matches
            }
        };
        let index = candidates[self.next.fetch_add(1, Ordering::Relaxed) % candidates.len()];
        let worker = &self.workers[index];
        let pending = PendingRequest::new(&worker.pending_requests);
        Some((index, worker, pending))
    }

    fn record_route(&self, prompt: &str, response: &[u8], worker: usize) {
        let mut tree = self.tree.lock().unwrap();
        if !self.workers[worker].health.lock().unwrap().healthy {
            return;
        }
        tree.insert(prompt, worker);
        if let Ok(text) = std::str::from_utf8(response) {
            tree.insert(text, worker);
        }
    }

    fn update_health(&self, index: usize, error: Option<String>) {
        let mut tree = self.tree.lock().unwrap();
        let mut health = self.workers[index].health.lock().unwrap();
        let was_healthy = health.healthy;
        *health = Health {
            healthy: error.is_none(),
            checked_at_ms: Some(now_ms()),
            error,
        };
        if was_healthy && !health.healthy {
            tree.remove_worker(index);
        }
    }

    fn has_healthy(&self) -> bool {
        self.workers
            .iter()
            .any(|worker| worker.health.lock().unwrap().healthy)
    }

    fn status_json(&self) -> Vec<u8> {
        let workers: Vec<_> = self
            .workers
            .iter()
            .map(|worker| {
                serde_json::json!({
                    "url": worker.url,
                    "health": worker.health.lock().unwrap().clone(),
                    "pending_requests": worker.pending_requests.load(Ordering::Relaxed),
                })
            })
            .collect();
        serde_json::to_vec(&serde_json::json!({ "workers": workers })).unwrap()
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn check_workers(cluster: &Cluster, agent: &ureq::Agent) {
    for (index, worker) in cluster.workers.iter().enumerate() {
        let result = agent.get(&format!("{}/health", worker.url)).call();
        let error = match result {
            Ok(response) if response.status() == 200 => None,
            Ok(response) => Some(format!("HTTP {}", response.status())),
            Err(error) => Some(error.to_string()),
        };
        cluster.update_health(index, error);
    }
}

fn reply(request: Request, status: u16, body: Vec<u8>, content_type: &str) {
    let response = Response::from_data(body)
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", content_type).unwrap());
    if let Err(error) = request.respond(response) {
        eprintln!("response error: {error}");
    }
}

fn handle_live(request: Request) {
    reply(request, 200, b"ok\n".to_vec(), "text/plain");
}

fn handle_ready(request: Request, cluster: &Cluster) {
    let status = if cluster.has_healthy() { 200 } else { 503 };
    let body = if status == 200 {
        b"ok\n".to_vec()
    } else {
        b"no healthy workers\n".to_vec()
    };
    reply(request, status, body, "text/plain");
}

fn handle_workers(request: Request, cluster: &Cluster) {
    reply(request, 200, cluster.status_json(), "application/json");
}

fn handle_generate(mut request: Request, cluster: &Cluster, agent: &ureq::Agent) {
    let mut prompt_body = Vec::new();
    let read = request.as_reader().take(8193).read_to_end(&mut prompt_body);
    if read.is_err() {
        reply(
            request,
            400,
            b"failed to read prompt\n".to_vec(),
            "text/plain",
        );
        return;
    }
    if prompt_body.len() > 8192 {
        reply(request, 413, b"prompt too large\n".to_vec(), "text/plain");
        return;
    }
    let prompt = std::str::from_utf8(&prompt_body).unwrap_or("");
    let Some((worker_index, worker, _pending)) = cluster.select_worker(prompt) else {
        reply(request, 503, b"no healthy workers\n".to_vec(), "text/plain");
        return;
    };
    let result = agent
        .post(&format!("{}/generate", worker.url))
        .send_bytes(&prompt_body);
    let response = match result {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        Err(error) => {
            cluster.update_health(worker_index, Some(error.to_string()));
            eprintln!("worker {}: {error}", worker.url);
            reply(request, 502, b"worker unavailable\n".to_vec(), "text/plain");
            return;
        }
    };
    let status = response.status();
    let content_type = response
        .header("Content-Type")
        .unwrap_or("text/plain")
        .to_owned();
    let mut body = Vec::new();
    if response.into_reader().read_to_end(&mut body).is_err() {
        reply(
            request,
            502,
            b"worker response failed\n".to_vec(),
            "text/plain",
        );
        return;
    }
    if (200..300).contains(&status) {
        cluster.record_route(prompt, &body, worker_index);
    }
    reply(request, status, body, &content_type);
}

fn handle(request: Request, cluster: &Cluster, agent: &ureq::Agent) {
    match (request.method(), request.url()) {
        (&Method::Get, "/live") => handle_live(request),
        (&Method::Get, "/ready") => handle_ready(request, cluster),
        (&Method::Get, "/workers") => handle_workers(request, cluster),
        (&Method::Post, "/generate") => handle_generate(request, cluster, agent),
        _ => reply(request, 404, b"not found\n".to_vec(), "text/plain"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/workers.yaml", env!("CARGO_MANIFEST_DIR")));
    let config: Config = serde_yaml::from_str(&fs::read_to_string(path)?)?;
    let cluster = Arc::new(Cluster::new(config.workers));
    let health_agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(500))
        .build();
    check_workers(&cluster, &health_agent);
    let health_cluster = Arc::clone(&cluster);
    let interval = Duration::from_millis(config.health_interval_ms.max(1));
    thread::spawn(move || loop {
        thread::sleep(interval);
        check_workers(&health_cluster, &health_agent);
    });

    let server = Server::http(&config.listen).map_err(|error| -> Box<dyn Error> { error })?;
    let agent = Arc::new(
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .build(),
    );
    eprintln!("Router listening on http://{}", config.listen);
    for request in server.incoming_requests() {
        let cluster = Arc::clone(&cluster);
        let agent = Arc::clone(&agent);
        thread::spawn(move || handle(request, &cluster, &agent));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_skips_unhealthy_workers() {
        let cluster = Cluster::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(
            cluster.tree.lock().unwrap().find("unknown").workers,
            &[0, 1, 2]
        );
        assert!(cluster.select_worker("unknown").is_none());
        cluster.workers[0].health.lock().unwrap().healthy = true;
        cluster.workers[2].health.lock().unwrap().healthy = true;
        let picked: Vec<_> = (0..4)
            .map(|_| cluster.select_worker("unknown").unwrap().1.url.as_str())
            .collect();
        assert_eq!(picked, ["a", "c", "a", "c"]);
    }

    #[test]
    fn prefix_selection_and_unhealthy_fallback() {
        let cluster = Cluster::new(vec!["a".into(), "b".into()]);
        for worker in &cluster.workers {
            worker.health.lock().unwrap().healthy = true;
        }
        cluster.record_route("Once upon a time", b"Once upon a time ago\n", 1);
        assert_eq!(cluster.select_worker("Once upon a hill").unwrap().0, 1);
        assert_eq!(
            cluster
                .select_worker("Once upon a time ago today")
                .unwrap()
                .0,
            1
        );

        cluster.workers[1].health.lock().unwrap().healthy = false;
        assert_eq!(cluster.select_worker("Once upon a hill").unwrap().0, 0);
    }

    #[test]
    fn response_continuation_selects_its_worker() {
        let cluster = Cluster::new(vec!["a".into(), "b".into()]);
        for worker in &cluster.workers {
            worker.health.lock().unwrap().healthy = true;
        }
        cluster.record_route("Once upon", b"Once upon a time\n", 1);
        cluster.record_route("Once upon", b"Once upon a hill\n", 0);

        assert_eq!(
            cluster.select_worker("Once upon a time again").unwrap().0,
            1
        );
        assert_eq!(
            cluster.select_worker("Once upon a hill again").unwrap().0,
            0
        );
    }

    #[test]
    fn unhealthy_worker_loses_prefixes_and_can_recover() {
        let cluster = Cluster::new(vec!["a".into(), "b".into()]);
        cluster.update_health(0, None);
        cluster.update_health(1, None);
        cluster.record_route("Once upon a time", b"Once upon a time ago", 1);

        cluster.update_health(1, Some("connection failed".into()));
        assert_eq!(
            cluster.tree.lock().unwrap().find("Once upon").matched_words,
            0
        );
        assert_eq!(cluster.select_worker("Once upon a time").unwrap().0, 0);
        cluster.record_route("Another story", b"Another story", 1);
        assert_eq!(
            cluster.tree.lock().unwrap().find("Another").matched_words,
            0
        );

        cluster.update_health(1, None);
        assert_eq!(cluster.tree.lock().unwrap().find("").workers, &[0, 1]);
        cluster.record_route("Once upon a time", b"Once upon a time ago", 1);
        assert_eq!(cluster.tree.lock().unwrap().find("Once upon").workers, &[1]);
    }

    #[test]
    fn pending_request_count_tracks_request_lifetime() {
        let cluster = Cluster::new(vec!["a".into()]);
        let worker = &cluster.workers[0];
        {
            let _first = PendingRequest::new(&worker.pending_requests);
            let _second = PendingRequest::new(&worker.pending_requests);
            let status: serde_json::Value = serde_json::from_slice(&cluster.status_json()).unwrap();
            assert_eq!(status["workers"][0]["pending_requests"], 2);
        }
        assert_eq!(worker.pending_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn switches_from_prefix_match_to_least_loaded() {
        let cluster = Cluster::new(vec!["a".into(), "b".into()]);
        cluster.update_health(0, None);
        cluster.update_health(1, None);
        cluster.record_route("Once upon a time", b"Once upon a time ago", 1);

        let _first = PendingRequest::new(&cluster.workers[1].pending_requests);
        assert_eq!(
            cluster.select_worker("Once upon a time again").unwrap().0,
            1
        );

        let _second = PendingRequest::new(&cluster.workers[1].pending_requests);
        let (index, _, pending) = cluster.select_worker("Once upon a time again").unwrap();
        assert_eq!(index, 0);
        assert_eq!(
            cluster.workers[0].pending_requests.load(Ordering::Relaxed),
            1
        );
        drop(pending);
        assert_eq!(
            cluster.workers[0].pending_requests.load(Ordering::Relaxed),
            0
        );
    }
}

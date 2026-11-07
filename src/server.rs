//! Synchronous request/reply plumbing for a future scheduled model worker.
//!
//! This module is intentionally not used by the current `serve` function yet.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tiny_http::{Request, Server as HttpServer};

pub type RequestId = u64;

/// The HTTP request stays on the server side while the model does its work.
pub struct PendingHttpRequest {
    pub request: Request,
    pub received_at: Instant,
}

/// The parsed work item sent to the model worker.
pub struct ModelRequest {
    pub id: RequestId,
    pub prompt: String,
    pub received_at: Instant,
}

/// A completed model result sent back to the HTTP reply worker.
pub struct ModelReply {
    pub id: RequestId,
    pub status: u16,
    pub body: Vec<u8>,
}

/// Owns the work queue, pending HTTP requests, and the two worker threads.
pub struct Server {
    pub http: HttpServer,
    pending: Arc<Mutex<HashMap<RequestId, PendingHttpRequest>>>,
    next_id: AtomicU64,
    work_tx: Sender<ModelRequest>,
    model_thread: JoinHandle<()>,
    reply_thread: JoinHandle<()>,
}

impl Server {
    /// Starts the model and reply threads with the supplied model worker loop.
    pub fn new<F>(http: HttpServer, model_worker: F) -> Self
    where
        F: FnOnce(Receiver<ModelRequest>, Sender<ModelReply>) + Send + 'static,
    {
        let pending = Arc::new(Mutex::new(HashMap::<RequestId, PendingHttpRequest>::new()));
        let (work_tx, work_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel::<ModelReply>();

        let reply_pending = Arc::clone(&pending);
        let reply_thread = thread::spawn(move || {
            for reply in reply_rx {
                let pending_request = reply_pending.lock().unwrap().remove(&reply.id);
                if let Some(pending_request) = pending_request {
                    if let Err(error) = pending_request
                        .request
                        .respond(crate::response(reply.body, reply.status))
                    {
                        eprintln!("response error: {error}");
                    }
                }
            }
        });
        let model_thread = thread::spawn(move || model_worker(work_rx, reply_tx));

        Self {
            http,
            pending,
            next_id: AtomicU64::new(1),
            work_tx,
            model_thread,
            reply_thread,
        }
    }

    /// Registers the HTTP request before enqueuing its model work so an early
    /// reply can always find the corresponding connection.
    pub fn enqueue(
        &self,
        request: Request,
        prompt: String,
    ) -> Result<RequestId, mpsc::SendError<ModelRequest>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let received_at = Instant::now();
        self.pending.lock().unwrap().insert(
            id,
            PendingHttpRequest {
                request,
                received_at,
            },
        );
        let work = ModelRequest {
            id,
            prompt,
            received_at,
        };
        if let Err(error) = self.work_tx.send(work) {
            let pending_request = self.pending.lock().unwrap().remove(&id);
            if let Some(pending_request) = pending_request {
                let _ = pending_request
                    .request
                    .respond(crate::response(b"model worker unavailable\n".to_vec(), 503));
            }
            return Err(error);
        }
        Ok(id)
    }

    /// Closes the work queue, then waits for both workers to finish.
    pub fn shutdown(self) -> thread::Result<()> {
        drop(self.work_tx);
        let model_result = self.model_thread.join();
        let reply_result = self.reply_thread.join();
        model_result?;
        reply_result
    }
}

//! Synchronous HTTP request/reply plumbing for the model worker.

use crate::model::Transformer;
use crate::request::generate;
use crate::tokenizer::Tokenizer;
use crate::Sampler;
use std::collections::HashMap;
use std::error::Error;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server as HttpServer, StatusCode};

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
    pub steps: usize,
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

fn response(body: impl Into<Vec<u8>>, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(body.into())
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap())
}

fn reply_worker(
    pending: Arc<Mutex<HashMap<RequestId, PendingHttpRequest>>>,
    reply_rx: Receiver<ModelReply>,
) {
    for reply in reply_rx {
        let pending_request = pending.lock().unwrap().remove(&reply.id);
        if let Some(pending_request) = pending_request {
            if let Err(error) = pending_request
                .request
                .respond(response(reply.body, reply.status))
            {
                eprintln!("response error: {error}");
            }
        }
    }
}

fn model_worker(
    mut model: Transformer,
    tokenizer: Tokenizer,
    mut sampler: Sampler,
    cache_chunk_size: usize,
    work_rx: Receiver<ModelRequest>,
    reply_tx: Sender<ModelReply>,
) {
    for work in work_rx {
        let body = generate(
            &mut model,
            &tokenizer,
            &mut sampler,
            &work.prompt,
            work.steps,
            cache_chunk_size,
        );
        if reply_tx
            .send(ModelReply {
                id: work.id,
                status: 200,
                body,
            })
            .is_err()
        {
            break;
        }
    }
}

impl Server {
    /// Starts the model and reply threads.
    pub fn new(
        port: u16,
        model: Transformer,
        tokenizer: Tokenizer,
        sampler: Sampler,
        cache_chunk_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let address = format!("127.0.0.1:{port}");
        let http = HttpServer::http(&address).map_err(|e| -> Box<dyn Error> { e })?;
        eprintln!("Listening on http://{address}");
        let pending = Arc::new(Mutex::new(HashMap::<RequestId, PendingHttpRequest>::new()));
        let (work_tx, work_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel::<ModelReply>();

        let reply_pending = Arc::clone(&pending);
        let reply_thread = thread::spawn(move || reply_worker(reply_pending, reply_rx));
        let model_thread = thread::spawn(move || {
            model_worker(
                model,
                tokenizer,
                sampler,
                cache_chunk_size,
                work_rx,
                reply_tx,
            )
        });

        Ok(Self {
            http,
            pending,
            next_id: AtomicU64::new(1),
            work_tx,
            model_thread,
            reply_thread,
        })
    }

    /// Accepts HTTP requests and sends valid generation work to the model thread.
    pub fn serve(&self, steps: usize) {
        for mut request in self.http.incoming_requests() {
            match (request.method(), request.url()) {
                (&Method::Get, "/health") => {
                    if let Err(error) = request.respond(response(b"ok\n".to_vec(), 200)) {
                        eprintln!("response error: {error}");
                    }
                }
                (&Method::Post, "/generate") => {
                    let mut body = Vec::new();
                    let mut limited = request.as_reader().take(8193);
                    let prompt = match limited.read_to_end(&mut body) {
                        Ok(_) if body.len() > 8192 => Err((413, b"prompt too large\n".to_vec())),
                        Ok(_) => match String::from_utf8(body) {
                            Ok(prompt) if !prompt.contains('\0') => Ok(prompt),
                            _ => Err((400, b"prompt must be UTF-8 without NUL\n".to_vec())),
                        },
                        Err(_) => Err((400, b"failed to read prompt\n".to_vec())),
                    };
                    match prompt {
                        Ok(prompt) => {
                            if let Err(error) = self.enqueue(request, prompt, steps) {
                                eprintln!("model worker unavailable: {error}");
                            }
                        }
                        Err((status, body)) => {
                            if let Err(error) = request.respond(response(body, status)) {
                                eprintln!("response error: {error}");
                            }
                        }
                    }
                }
                _ => {
                    if let Err(error) = request.respond(response(b"not found\n".to_vec(), 404)) {
                        eprintln!("response error: {error}");
                    }
                }
            }
        }
    }

    /// Registers the HTTP request before enqueuing its model work so an early
    /// reply can always find the corresponding connection.
    pub fn enqueue(
        &self,
        request: Request,
        prompt: String,
        steps: usize,
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
            steps,
        };
        if let Err(error) = self.work_tx.send(work) {
            let pending_request = self.pending.lock().unwrap().remove(&id);
            if let Some(pending_request) = pending_request {
                let _ = pending_request
                    .request
                    .respond(response(b"model worker unavailable\n".to_vec(), 503));
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

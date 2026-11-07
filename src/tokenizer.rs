use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    lookup: HashMap<Vec<u8>, usize>,
}

fn read_i32(file: &mut File) -> io::Result<i32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

impl Tokenizer {
    pub fn load(path: &Path, vocab_size: usize) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let max_token_length = read_i32(&mut file)?;
        if max_token_length <= 0 || max_token_length > 1_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid tokenizer header",
            ));
        }
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut lookup = HashMap::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let mut score = [0; 4];
            file.read_exact(&mut score)?;
            scores.push(f32::from_le_bytes(score));
            let len = read_i32(&mut file)?;
            if len < 0 || len > max_token_length {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid token length",
                ));
            }
            let mut piece = vec![0; len as usize];
            file.read_exact(&mut piece)?;
            lookup.insert(piece.clone(), id);
            vocab.push(piece);
        }
        Ok(Self {
            vocab,
            scores,
            lookup,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(1); // BOS
        if !text.is_empty() {
            tokens.push(
                *self
                    .lookup
                    .get(b" ".as_slice())
                    .expect("missing space token"),
            );
        }
        for ch in text.chars() {
            let mut bytes = [0; 4];
            let piece = ch.encode_utf8(&mut bytes).as_bytes();
            if let Some(&id) = self.lookup.get(piece) {
                tokens.push(id);
            } else {
                tokens.extend(piece.iter().map(|&byte| byte as usize + 3));
            }
        }
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let mut pair = self.vocab[tokens[i]].clone();
                pair.extend_from_slice(&self.vocab[tokens[i + 1]]);
                if let Some(&id) = self.lookup.get(&pair) {
                    let score = self.scores[id];
                    if best.is_none_or(|(_, _, current)| score > current) {
                        best = Some((i, id, score));
                    }
                }
            }
            match best {
                Some((i, id, _)) => {
                    tokens[i] = id;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }
        tokens
    }

    pub fn decode(&self, previous: usize, token: usize) -> Vec<u8> {
        let mut piece = self.vocab[token].as_slice();
        if previous == 1 && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        if piece.len() == 6 && piece.starts_with(b"<0x") && piece.ends_with(b">") {
            if let Ok(hex) = std::str::from_utf8(&piece[3..5]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    piece = std::slice::from_ref(&byte);
                    return safe_piece(piece);
                }
            }
        }
        safe_piece(piece)
    }
}

fn safe_piece(piece: &[u8]) -> Vec<u8> {
    if piece.len() == 1 && !piece[0].is_ascii_graphic() && !piece[0].is_ascii_whitespace() {
        Vec::new()
    } else {
        piece.to_vec()
    }
}

use llama2_rs::benchmark::benchmark_forward;
use llama2_rs::model::Transformer;
use std::env;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let checkpoint = args
        .next()
        .ok_or("usage: bench_forward <checkpoint> [batch_size] [steps] [repeats]")?;
    let batch_size = args.next().map(|s| s.parse()).transpose()?.unwrap_or(4);
    let steps = args.next().map(|s| s.parse()).transpose()?.unwrap_or(8);
    let repeats = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);
    let model = Transformer::load(Path::new(&checkpoint))?;
    println!("{}", benchmark_forward(&model, batch_size, steps, repeats));
    Ok(())
}

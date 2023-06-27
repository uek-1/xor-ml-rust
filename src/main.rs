mod model;
use model::{Model, Layer, Activation};
use rand::{Rng, rngs::ThreadRng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let numb_gen = |rng : &mut ChaCha8Rng| rng.gen_range((0.5)..(1.0));

    let mut layers : Vec<Layer> = vec![];
    let weights = vec![vec![numb_gen(&mut rng), numb_gen(&mut rng)], vec![numb_gen(&mut rng), numb_gen(&mut rng)]];
    let biases = vec![0.0, 0.0];
    let hidden_layer_1 = Layer::new(weights,biases, Activation::Sigmoid);
    layers.push(hidden_layer_1);
    
    let weights = vec![vec![numb_gen(&mut rng), numb_gen(&mut rng)]];
    let biases = vec![0.0,0.0];
    let output_layer = Layer::new(weights,biases, Activation::None);
    layers.push(output_layer);

    let model = Model::from(layers);
 
    println!("{:?}", model);
    
    let mut training_data : Vec<(Vec<f64>,  f64)> = vec![];

    for x in 0..2 {
        for y in 0..2 {
            training_data.push((vec![(x) as f64, (y) as f64], (x ^ y) as f64))
        }
    }
    
    println!("{:?}", training_data);

    let model = model.to_trained(training_data, 500, 1.2);
    
    println!("{:?}", model);

    let input_1 = vec![0.0,0.0];
    let res_1 = model.evaluate(&input_1);
    let test_cost_1 = model.cost(&input_1, 0.0);
    println!("0 XOR 0  = {res_1}, test cost = {test_cost_1}");   
    
    let input_2 = vec![1.0,0.0];
    let res_2 = model.evaluate(&input_2);
    let test_cost_2 = model.cost(&input_2, 1.0);
    println!("1 XOR 0  = {res_2} test cost = {test_cost_2}");

    let input_3 = vec![1.0,1.0];
    let res_3 = model.evaluate(&input_3);
    let test_cost_3 = model.cost(&input_3, 0.0);
    println!("1 XOR 1  = {res_3} test cost = {test_cost_3}");

}

fn xor(x : f64, y : f64) -> f64 {
    if x.floor() + y.floor() == 1.0 {
        return 1.0
    }

    0.0
}


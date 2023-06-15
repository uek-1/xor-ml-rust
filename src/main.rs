mod model;
use model::{Model, Layer, Activation};

fn main() {
    let mut layers : Vec<Layer> = vec![];
    let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let biases = vec![0.0, 0.0];
    let hidden_layer_1 = Layer::new(weights,biases, Activation::Sigmoid);
    layers.push(hidden_layer_1);
    
    let weights = vec![vec![1.0,1.0]];
    let biases = vec![0.0,0.0];
    let output_layer = Layer::new(weights,biases, Activation::None);
    layers.push(output_layer);

    let model = Model::from(layers);
 
    let input = vec![1.0,1.0];
    let res = model.evaluate(&input);

    println!("{res}");   
    
    let mut training_data : Vec<(Vec<f64>,  f64)> = vec![];

    for x in 0..2 {
        for y in 0..2 {
            training_data.push((vec![(x % 2) as f64, (y % 2) as f64], xor( (x % 2) as f64, (y % 2) as f64)))
        }
    }

    dbg!(training_data.clone());

    let model = model.to_trained(training_data, 5, 0.2);
    
    let input_1 = vec![0.0,0.0];
    let res_1 = model.evaluate(&input_1);
    println!("0 XOR 0  = {res_1}");   
    
    let input_2 = vec![1.0,0.0];
    let res_2 = model.evaluate(&input_2);
    println!("1 XOR 0  = {res_2}");

    let input_3 = vec![1.0,1.0];
    let res_3 = model.evaluate(&input_3);
    println!("1 XOR 1  = {res_3}");
}

fn xor(x : f64, y : f64) -> f64 {
    if x.floor() + y.floor() == 1.0 {
        return 1.0
    }

    0.0
}


mod model;
use model::{Model, Layer};

fn main() {
    let mut layers : Vec<Layer> = vec![];
    let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let biases = vec![0.0, 0.0];
    let hidden_layer_1 = Layer::new(weights,biases);
    layers.push(hidden_layer_1);
    
    let weights = vec![vec![1.0,1.0]];
    let biases = vec![0.0,0.0];
    let output_layer = Layer::new(weights,biases);
    layers.push(output_layer);

    let model = Model::from(layers);
 
    let input = vec![1.0,1.0];
    let res = model.evaluate(&input);

    println!("{res}");   
    
    let mut training_data : Vec<(Vec<f64>,  f64)> = vec![];

    for i in 0..5 {
        training_data.push((vec![i as f64, i as f64], 15.0 * i as f64))
    }

    let model = model.to_trained(training_data, 3);

}
  


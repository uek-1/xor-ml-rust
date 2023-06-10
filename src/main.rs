mod model;
use model::{Model, Layer};

fn main() {
    let mut layers : Vec<Layer> = vec![];
    let weights = vec![vec![1, 0], vec![0, 1]];
    let biases = vec![0, 0];
    let hidden_layer_1 = Layer::new(weights,biases);
    layers.push(hidden_layer_1);
    
    let weights = vec![vec![1,1]];
    let biases = vec![0,0];
    let output_layer = Layer::new(weights,biases);
    layers.push(output_layer);

    let model = Model::from(layers);
 
    let input = vec![1,1];
    let res = model.evaluate(&input);

    println!("{res}");   

    let model = train(model, 3);

}
  
fn train(model: Model, epochs: usize) -> Model {
    let training_data : Vec<(Vec<u8>,  u8)> = vec![
        (vec![1, 1], 0),
        (vec![1, 0], 1),
        (vec![0, 0], 0)
    ];

    for i in 0..epochs {
        println!("EPOCH {i}");
        let mut total_cost : u8 = 0;

        for (input, output) in &training_data {
            total_cost += cost(&model, &input, *output);       
        }
        println!("TOTAL COST {total_cost}");
    }

    Model::new()
    //single hidden layer. 
}

fn cost(model : &Model, input : &Vec<u8>, expected : u8) -> u8 {
    (expected as i32 - model.evaluate(input) as i32).abs() as u8
 }


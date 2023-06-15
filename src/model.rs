use std::ops::{DerefMut, Deref};

#[derive(Clone, PartialEq)]
pub enum Activation {
    None,
    Relu,
}

impl Activation {
    pub fn process_vec(input : Vec<f64>, activation : &Activation) -> Vec<f64> {
        input
            .iter()
            .map(|x| Activation::process_num(*x, activation))
            .collect()
    }

    pub fn process_num(input : f64, activation : &Activation) -> f64 {
        match activation {
            Activation::None => input,
            Activation::Relu => Self::relu(input),
        }
    }

    pub fn relu(input : f64) -> f64 {
        match input > 0.0 {
            true => input,
            false => 0.0
        }
    }
}


#[derive(Clone)]
pub struct Layer{
    pub weights : Vec<Vec<f64>>,
    pub biases : Vec<f64>,
    pub activation : Activation,
}

impl Layer {
    pub fn new(weights : Vec<Vec<f64>>, biases : Vec<f64>, activation : Activation) -> Layer {
        Layer {
            weights,
            biases,
            activation
        }
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        let out : Vec<f64> = matrix_column_multiply(&self.weights, &input).add_vec(&self.biases);
        Activation::process_vec(out, &self.activation)
    }
}

#[derive(Clone)]
pub struct Model(Vec<Layer>);

impl Model {
    pub fn new() -> Model {
        Model(vec![])
    }

    pub fn with_layer(&self, layer : Layer) -> Model {
        let mut layers = self.0.clone();
        layers.push(layer);
        Model(layers)
    }

    pub fn to_updated(&self, layer : Layer, layer_index : usize) -> Result<Model, &'static str> {
        if self.len() <= layer_index {
            return Err("invalid index!");
        }
        let mut updated  : Model = self.clone();
        updated[layer_index] = layer;
        Ok(updated)
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> f64 {
        *self
            .iter()
            .fold(input.clone(), |temp, layer| layer.evaluate(&temp))
            .get(0)
            .unwrap()
    }

    pub fn to_trained(&self, training_data : Vec<(Vec<f64>, f64)>, epochs: usize) -> Model {
        let mut trained : Model = self.clone();
        println!("LAYERS : {}" ,trained.len());

        for i in 0..epochs {
            println!("EPOCH {i}");

            dbg!(trained[0].weights.clone());

            let mut average_cost : f64 = 0.0;

            for (input, output) in &training_data {
                let epoch_cost = trained.cost(&input, *output);
                let current_weights = &trained[0].weights;
                let current_biases = trained[0].biases.clone();

                let mut new_weights : Vec<Vec<f64>> = vec![vec![0.0 ; 2]; 2];
                let mut new_biases : Vec<Vec<f64>> = vec![vec![0.0 ; 2]; 2];
                
                let layer_eval = self[0].evaluate(input);
                
                for (num, neuron) in current_weights.iter().enumerate() {
                    new_weights[num][0] = neuron[0] - (Activation::relu(input[0]) * input[0]) * (2.0 * layer_eval[0] + 2.0 * layer_eval[1] - 2.0 * output) * 0.003; 
                    new_weights[num][1] = neuron[1] - (Activation::relu(input[1]) * input[1]) * (2.0 * layer_eval[0] + 2.0 * layer_eval[1] - 2.0 * output) * 0.003;
                } 

                trained = trained.to_updated(Layer::new(new_weights, current_biases, self[0].activation.clone()), 0).unwrap();

                average_cost += trained.cost(&input, *output);     
            }

            average_cost = average_cost / training_data.len() as f64; 
            println!("Average Cost {average_cost}");
        }
        
        trained
    }

    fn cost(&self, input : &Vec<f64>, expected : f64) -> f64 {
        (expected - self.evaluate(input)) *  (expected - self.evaluate(input))
    }


}

impl From<Vec<Layer>> for Model {
    fn from(layers : Vec<Layer>) -> Model {
        Model(layers)
    }
}

impl Deref for Model {
    type Target = Vec<Layer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Model {
    fn deref_mut(&mut self) -> &mut Vec<Layer> {
        &mut self.0
    }
}

impl AddVec<f64> for Vec<f64> {
    fn add_vec(&self, other : &Vec<f64>) -> Vec<f64>{
        self
            .iter()
            .zip(other.iter())
            .map(|(x,y)| x + y)
            .collect()
    }
}

trait AddVec<T> {
    fn add_vec(&self, other : &Vec<T>) -> Vec<T>;
}

fn dot_product(left : &Vec<f64>, right : &Vec<f64>) -> f64 {
    left
        .iter()
        .zip(right.iter())
        .fold(0.0, |acc, (x,y)| acc + x * y)
}

fn matrix_column_multiply(mat : &Vec<Vec<f64>>, col : &Vec<f64>) -> Vec<f64> {
    mat
        .iter()
        .map(|row| dot_product(row, col))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot_prod_test() {
        let x : Vec<f64> = vec![1.0, 2.0];
        let y : Vec<f64> = vec![2.0, 4.0];
        let res = dot_product(&x, &y);

        assert_eq!(res, 10.0);
    }

    #[test]
    fn matrix_column_multiply_test() {
        let mat : Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let col : Vec<f64> = vec![2.0, 3.0];
        let res = matrix_column_multiply(&mat, &col);
        assert_eq!(res, vec![2.0,3.0]);
    }

    #[test]
    fn relu_test_1() {
        let res = Activation::relu(-2.0);
        assert_eq!(res, 0.0)
    }

    #[test]
    fn relu_test_2() {
        let res = Activation::relu(2.0);
        assert_eq!(res, 2.0)
    }
}

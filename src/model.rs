use std::ops::{DerefMut, Deref};

#[derive(Clone, PartialEq)]
pub enum Activation {
    None,
    Relu,
    Sigmoid,
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
            Activation::Sigmoid => Self::sigmoid(input)
        }
    }

    pub fn relu(input : f64) -> f64 {
        match input > 0.0 {
            true => input,
            false => 0.0
        }
    }

    pub fn sigmoid(input : f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
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

    fn forward_prop(&self, input : Vec<f64>) -> (Vec<f64>,f64) {
        let a1 = self[0].evaluate(&input);
        let a2 = self[0].evaluate(&a1)[0];
        
        (a1,a2)
    }

    fn backward_prop(&self, a1 : Vec<f64>, a2 : f64, input: Vec<f64>, expected : f64) -> (Vec<Vec<f64>>, Vec<f64>){
        let da_2 = 2.0 * (a2 - expected); 
        let dz_2 = da_2 * (a2 * (1.0-a2));
        let dw_2 : Vec<f64> = a1.iter().map(|x| x * dz_2).collect();

        let da_1 : Vec<f64> = self[1].weights[0].iter().map(|x| x * dz_2).collect();
        let deriv_sig_z1 : Vec<f64> = a1.iter().map(|x| x * (1.0 - x)).collect();
        let dz_1 = da_1.iter()
            .zip(deriv_sig_z1.iter())
            .fold(0.0, |acc, (x,y)| acc + x * y);

        let dw_1 = vec![vec![dz_1 * input[0], dz_1 * input[1]]; 2];

        (dw_1, dw_2)
    }

    pub fn to_trained(&self, training_data : Vec<(Vec<f64>, f64)>, epochs: usize, rate: f64) -> Model {
        let mut trained : Model = self.clone();
        println!("LAYERS : {}" ,trained.len());

        for i in 0..epochs {
            println!("EPOCH {i}");

            dbg!(trained[0].weights.clone());

            let mut average_cost : f64 = 0.0;

            for (input, output) in &training_data {                
                let (a1, a2) = trained.forward_prop(input.to_vec());
                let (dw_1, dw_2) = trained.backward_prop(a1, a2, input.to_vec(), *output);

                let scaled_dw_1 : Vec<Vec<f64>> = dw_1.iter()
                    .map(|row| row
                        .iter()
                        .map(|x| x * rate)
                        .collect()
                    )
                    .collect();

                let scaled_dw_2 : Vec<f64> = dw_2.iter()
                    .map(|x| x * rate)
                    .collect();

                let mut w_1 : Vec<Vec<f64>> = scaled_dw_1;
                let mut w_2 : Vec<Vec<f64>> = vec![scaled_dw_2];

                for i in 0..w_1.len() {
                    for j in 0..w_1[i].len() {
                        w_1[i][j] = trained[0].weights[i][j] - w_1[i][j];
                     }
                }

                for i in 0..w_2.len() {
                    w_2[0][i] = trained[1].weights[0][i] - w_2[0][i];
                }
                
                trained = trained.to_updated(Layer::new(w_1, trained[0].biases.clone(), trained[0].activation.clone()), 0).unwrap();
                trained = trained.to_updated(Layer::new(w_2, trained[1].biases.clone(), trained[1].activation.clone()), 1).unwrap();

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

use std::ops::{DerefMut, Deref};

#[derive(Clone, PartialEq, Debug)]
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


#[derive(Clone, Debug)]
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

    pub fn neurate(&self, input: &Vec<f64>) -> Vec<f64> {
        matrix_column_multiply(&self.weights, &input)
    }
}

#[derive(Clone, Debug)]
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

    fn forward_prop(&self, input : Vec<f64>) -> (Vec<f64>, f64, Vec<f64>,f64) {
        let a1 = self[0].evaluate(&input);
        let z1 = self[0].neurate(&input);
        let a2 = self[1].evaluate(&a1)[0];
        let z2 = self[1].neurate(&a1)[0];
       
        //assert_eq!(a2, z2);
        (z1, z2, a1,a2)
    }

    pub fn backward_prop(&self, z1 : Vec<f64>, z2: f64, activation_1 : Vec<f64>, activation_2 : f64, input: Vec<f64>, expected : f64) -> (Vec<Vec<f64>>, Vec<f64>){
        // 1 x 1
        let partial_loss_activation_2 = -2.0 * (expected - activation_2); 
        let partial_loss_neuron_2 = partial_loss_activation_2 * 1.0;
        // 1 x 2
        let partial_loss_weights_2 : Vec<f64> = activation_1.iter().map(|x| x * partial_loss_neuron_2).collect();
        
        // 1 x 2
        let partial_loss_activation_1 : Vec<f64> = self[1].weights[0].iter().map(|w_2| partial_loss_neuron_2 * w_2).collect();
        
        // Sigmoid derivative
        let partial_activation_neuron_1 : Vec<f64> = activation_1.iter().map(|a1| a1 * (1.0 - a1)).collect();

        let partial_loss_neuron_1 : Vec<f64> = partial_loss_activation_1.iter()
            .zip(partial_activation_neuron_1.iter())
            .map(|(x,y)| x * y)
            .collect();
        
        // 2 x 2
        let partial_loss_weights_1 = partial_loss_neuron_1
            .iter()
            .map(
                |neuron| input.iter().map(|x| x * neuron).collect() 
            )
            .collect();

        (partial_loss_weights_1, partial_loss_weights_2)
    }

    fn debug_gradient_check(&self, dw_1 : Vec<Vec<f64>>, dw_2 : Vec<f64>, input : Vec<f64>, output: f64) {
        let epsilon = 0.0001;
        for row in 0..dw_1.len() {
            for col in 0..dw_1[row].len() {
                let weight = dw_1[row][col] ;
                let mut increment = self[0].clone();
                increment.weights[row][col] += epsilon;
                
                let mut decrement = self[0].clone();
                decrement.weights[row][col] -= epsilon;

                let model_increment = self.clone().to_updated(increment, 0).unwrap();
                let model_decrement = self.clone().to_updated(decrement, 0).unwrap(); 
                let res = (model_increment.cost(&input, output) - model_decrement.cost(&input, output)) / (2.0 * epsilon);
                assert!((weight - res).abs() < epsilon);
            }
        }

        for col in 0..dw_2.len() {
            let weight = dw_2[col];
            let mut increment = self[1].clone();
            increment.weights[0][col] += epsilon;
            
            let mut decrement = self[1].clone();
            decrement.weights[0][col] -= epsilon;

            let model_increment = self.clone().to_updated(increment, 1).unwrap();
            let model_decrement = self.clone().to_updated(decrement, 1).unwrap(); 
            let res = (model_increment.cost(&input, output) - model_decrement.cost(&input, output)) / (2.0 * epsilon);
            assert!((weight - res).abs() < epsilon);
        }
    }

    pub fn to_trained(&self, training_data : Vec<(Vec<f64>, f64)>, epochs: usize, rate: f64) -> Model {
        let mut trained : Model = self.clone();
        println!("LAYERS : {}" ,trained.len());

        for i in 0..epochs {
            println!("EPOCH {i}");
            
            //println!("{:?}", trained);
            
            let mut average_cost : f64 = 0.0;

            for (input, output) in &training_data {                
                let (z1, z2, a1, a2) = trained.forward_prop(input.to_vec());
                let (dw_1, dw_2) = trained.backward_prop(z1, z2, a1, a2, input.to_vec(), *output);
                
                trained.debug_gradient_check(dw_1.clone(), dw_2.clone(), input.clone(), output.clone());

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
                
                trained = trained
                    .to_updated(Layer::new(w_1, trained[0].biases.clone(), trained[0].activation.clone()), 0).unwrap()
                    .to_updated(Layer::new(w_2, trained[1].biases.clone(), trained[1].activation.clone()), 1).unwrap();

                average_cost += trained.cost(&input, *output);     
            }

            average_cost = average_cost / training_data.len() as f64; 
            println!("Average Cost {average_cost}");
        }
        
        trained
    }

    pub fn cost(&self, input : &Vec<f64>, expected : f64) -> f64 {
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
    fn matrix_column_multiply_test_2() {
        let mat : Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let col : Vec<f64> = vec![5.0, 6.0];
        let res = matrix_column_multiply(&mat, &col);
        assert_eq!(res, vec![17.0, 39.0])
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

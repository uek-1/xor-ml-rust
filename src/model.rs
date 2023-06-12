#[derive(Clone)]
pub struct Layer{
    weights : Vec<Vec<f64>>,
    biases : Vec<f64>,
}

impl Layer {
    pub fn new(weights : Vec<Vec<f64>>, biases : Vec<f64>) -> Layer {
        Layer {
            weights,
            biases
        }
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        matrix_column_multiply(&self.weights, &input).add_vec(&self.biases)
    }
}

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

    pub fn evaluate(&self, input: &Vec<f64>) -> f64 {
        *self.0
            .iter()
            .fold(input.clone(), |temp, layer| layer.evaluate(&temp))
            .get(0)
            .unwrap()
    }

    pub fn to_trained(&self, training_data : Vec<(Vec<f64>, f64)>, epochs: usize) -> Model {


        for i in 0..epochs {
            println!("EPOCH {i}");
            let mut total_cost : f64 = 0.0;

            for (input, output) in &training_data {
                total_cost += self.cost(&input, *output);       
            }
            println!("TOTAL COST {total_cost}");
        }

        Model::new()
        //single hidden layer. 
    }

    fn cost(&self, input : &Vec<f64>, expected : f64) -> f64 {
        (expected - self.evaluate(input)).abs()
    }


}

impl From<Vec<Layer>> for Model {
    fn from(layers : Vec<Layer>) -> Model {
        Model(layers)
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
}

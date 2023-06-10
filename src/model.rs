#[derive(Clone)]
pub struct Layer{
    weights : Vec<Vec<u8>>,
    biases : Vec<u8>,
}

impl Layer {
    pub fn new(weights : Vec<Vec<u8>>, biases : Vec<u8>) -> Layer {
        Layer {
            weights,
            biases
        }
    }

    pub fn evaluate(&self, input: &Vec<u8>) -> Vec<u8> {
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

    pub fn evaluate(&self, input: &Vec<u8>) -> u8 {
        *self.0
            .iter()
            .fold(input.clone(), |temp, layer| layer.evaluate(&temp))
            .get(0)
            .unwrap()
    }
}

impl From<Vec<Layer>> for Model {
    fn from(layers : Vec<Layer>) -> Model {
        Model(layers)
    }
}

impl AddVec<u8> for Vec<u8> {
    fn add_vec(&self, other : &Vec<u8>) -> Vec<u8>{
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

fn dot_product(left : &Vec<u8>, right : &Vec<u8>) -> u8 {
    left
        .iter()
        .zip(right.iter())
        .fold(0, |acc, (x,y)| acc + x * y)
}

fn matrix_column_multiply(mat : &Vec<Vec<u8>>, col : &Vec<u8>) -> Vec<u8> {
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
        let x : Vec<u8> = vec![1,2];
        let y : Vec<u8> = vec![2,4];
        let res = dot_product(&x, &y);

        assert_eq!(res, 10);
    }

    #[test]
    fn matrix_column_multiply_test() {
        let mat : Vec<Vec<u8>> = vec![vec![1, 0], vec![0, 1]];
        let col : Vec<u8> = vec![2,3];
        let res = matrix_column_multiply(&mat, &col);
        assert_eq!(res, vec![2,3]);
    }
}

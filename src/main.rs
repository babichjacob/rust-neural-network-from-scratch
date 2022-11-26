extern crate blas_src;

use ndarray::{prelude::*, OwnedRepr};

type Data = f64;

// struct Neuron<const INPUTS: usize> {
//     weights: [Data; INPUTS],
//     bias: Data,
// }

// impl<const INPUTS: usize> Neuron<INPUTS> {
//     fn predict(&self, inputs: &[Data; INPUTS]) -> Data {
//         arr1(&self.weights).dot(&arr1(inputs)) + self.bias
//     }
// }

// struct Layer<const INPUTS: usize, const NEURONS: usize> {
//     neurons: [Neuron<INPUTS>; NEURONS],
// }

// impl<const INPUTS: usize, const NEURONS: usize> Layer<INPUTS, NEURONS> {
//     fn predict(&self, inputs: &[Data; INPUTS]) -> Array1<f64> {
//         let predictions = self.neurons.map(|neuron| neuron.predict(inputs));
//         // let predictions: Vec<f64> = predictions.collect();
//         arr1(&predictions)
//     }
// }

struct DenseLayer {
    weights: Array2<Data>,
    biases: Array1<Data>,
}

// impl DenseLayer {
//     fn predict(&self, input: &Array1<Data>) -> Array1<Data> {
//         self.weights.dot(input) + &self.biases
//     }
// }

trait Layer {
    fn predict(&self, inputs: &Array2<Data>) -> Array2<Data>;
}

impl Layer for DenseLayer {
    fn predict(&self, inputs: &Array2<Data>) -> Array2<Data> {
        self.weights.dot(inputs) + &self.biases
    }
}

impl DenseLayer {
    fn new(neurons: usize, input: usize) -> Self {
        DenseLayer {
            weights: Array2::default((neurons, input)),
            biases: Array1::default((neurons,)),
        }
    }
}

struct SomeKindOfNetwork {
    hidden_layers: Vec<Box<dyn Layer>>,
}

impl SomeKindOfNetwork {
    fn new(hidden_layers: usize, neurons: usize, input: usize) -> Self {
        let layers = (0..hidden_layers)
            .map(|_| Box::new(DenseLayer::new(neurons, input)) as Box<dyn Layer>)
            .collect();

        SomeKindOfNetwork {
            hidden_layers: layers,
        }
    }
}

trait Network {
    fn predict(&self, inputs: &Array2<Data>) -> Array2<Data>;
}

fn softmax(vector: Array1<Data>) -> Array1<Data> {
    let exponentiated = vector.mapv(Data::exp);

    exponentiated.clone() / exponentiated.sum()
}

impl Network for SomeKindOfNetwork {
    fn predict(&self, inputs: &Array2<Data>) -> Array2<Data> {
        let mut layers = self.hidden_layers.iter();

        let first_layer = layers
            .next()
            .expect("neural networks in this implementation must have at least 1 hidden layer");

        let mut prediction = first_layer.predict(inputs);

        for layer in layers {
            prediction = layer.predict(&prediction);
        }

        let output: Vec<Data> = prediction
            .rows()
            .into_iter()
            .flat_map(|row| softmax(row.to_owned()))
            .collect();
        
        let output = Array::from_shape_vec(() output);

        output
    }
}

fn main() {
    println!("Hello, world!");

    let network = SomeKindOfNetwork::new(2, 64, 28 * 28);
}

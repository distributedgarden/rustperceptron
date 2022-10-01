use std::env;
use std::f64::consts::E;

fn activity_function(weights: &Vec<f64>, inputs: &Vec<f64>) -> f64 {
    let mut summation = 0.0;
    for index in 0..weights.len() {
        summation += weights[index] * inputs[index];
    }

    return summation;
}

fn activation_function(value: f64) -> f64 {
    let result: f64 = 1.0 / (1.0 + E.powf(-value));

    return result;
}

fn weight_function(
    weights: Vec<f64>,
    eta: f64,
    target: f64,
    activation_output: f64,
    inputs: &Vec<f64>,
) -> Vec<f64> {
    let mut updated_weights: Vec<f64> = vec![0.0; weights.len()];
    let end: usize = weights.len() - 1;

    for index in 0..end {
        let weight_delta: f64 = (eta * (target - activation_output)) * inputs[index];
        let weight: f64 = weights[index] + weight_delta;

        updated_weights[index] = weight;
    }

    let bias: f64 = weights[end] + (target - activation_output);
    updated_weights[end] = bias;

    return updated_weights;
}

fn perceptron(inputs: &Vec<f64>, target: f64, eta: f64, mut weights: Vec<f64>) -> Vec<f64> {
    let activity_output: f64 = activity_function(&weights, &inputs);
    let activation_output: f64 = activation_function(activity_output);

    println!(
        "target: {}, activity_output: {}, activation_output: {}",
        target, activity_output, activation_output
    );
    println!("weights: {:?}", &weights);
    if activation_output != target {
        weights = weight_function(weights, eta, target, activation_output, &inputs);
    }

    return weights;
}

fn perceptron_learning_algorithm(
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<f64>,
    eta: f64,
    mut weights: Vec<f64>,
    epoch: i32,
) -> Vec<f64> {
    for iteration in 0..epoch {
        println!("[iteration {}]", iteration);

        let end: usize = targets.len();
        for index in 0..end {
            weights = perceptron(&inputs[index], targets[index], eta, weights);
        }
    }

    return weights;
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let learn = &args[1];

    if learn == "single" {
        // includes additional value for bias
        let inputs: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 1.0]];
        let targets: Vec<f64> = vec![0.8];
        let eta: f64 = 0.01;
        let epoch: i32 = 20;

        // includes [w0, w1, bias]
        let mut weights: Vec<f64> = vec![1.0, 1.0, 0.01];

        weights = perceptron_learning_algorithm(&inputs, &targets, eta, weights, epoch);
        println!("{:?}", &weights);
    }

    if learn == "or" {
        // includes additional value for bias
        let inputs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let targets: Vec<f64> = vec![0.0, 1.0, 1.0, 1.0];
        let eta: f64 = 0.01;
        let epoch: i32 = 50;

        // includes [w0, w1, bias]
        let mut weights: Vec<f64> = vec![1.0, 1.0, 0.01];

        weights = perceptron_learning_algorithm(&inputs, &targets, eta, weights, epoch);
        println!("{:?}", &weights);
    }
}

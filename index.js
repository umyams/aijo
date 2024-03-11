// Define a simple neural network class
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Initialize weights
        this.weightsInputHidden = new Matrix(this.hiddenNodes, this.inputNodes);
        this.weightsHiddenOutput = new Matrix(this.outputNodes, this.hiddenNodes);
        this.weightsInputHidden.randomize();
        this.weightsHiddenOutput.randomize();

        // Initialize biases
        this.biasHidden = new Matrix(this.hiddenNodes, 1);
        this.biasOutput = new Matrix(this.outputNodes, 1);
        this.biasHidden.randomize();
        this.biasOutput.randomize();
    }

    // Feedforward function
    feedforward(inputArray) {
        // Generating the hidden outputs
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(sigmoid);

        // Generating the final output
        let output = Matrix.multiply(this.weightsHiddenOutput, hidden);
        output.add(this.biasOutput);
        output.map(sigmoid);

        return output.toArray();
    }

    // Training function using backpropagation
    train(inputArray, targetArray) {
        // Feedforward
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(sigmoid);

        let outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
        outputs.add(this.biasOutput);
        outputs.map(sigmoid);

        // Convert targets to matrix
        let targets = Matrix.fromArray(targetArray);

        // Calculate output errors
        let outputErrors = Matrix.subtract(targets, outputs);

        // Calculate hidden layer errors
        let hiddenErrors = Matrix.transpose(this.weightsHiddenOutput);
        hiddenErrors = Matrix.multiply(hiddenErrors, outputErrors);

        // Adjust weights and biases
        let gradients = Matrix.map(outputs, sigmoidDerivative);
        gradients.multiply(outputErrors);
        gradients.multiply(LEARNING_RATE);

        let hiddenT = Matrix.transpose(hidden);
        let weightsHiddenOutputDeltas = Matrix.multiply(gradients, hiddenT);

        this.weightsHiddenOutput.add(weightsHiddenOutputDeltas);
        this.biasOutput.add(gradients);

        // Calculate hidden gradients
        let hiddenGradients = Matrix.map(hidden, sigmoidDerivative);
        hiddenGradients.multiply(hiddenErrors);
        hiddenGradients.multiply(LEARNING_RATE);

        let inputsT = Matrix.transpose(inputs);
        let weightsInputHiddenDeltas = Matrix.multiply(hiddenGradients, inputsT);

        this.weightsInputHidden.add(weightsInputHiddenDeltas);
        this.biasHidden.add(hiddenGradients);
    }
}

// Sigmoid activation function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Derivative of sigmoid function
function sigmoidDerivative(x) {
    return x * (1 - x);
}

// Define a Matrix class for matrix operations
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }

    // Function to apply a function to every element of the matrix
    map(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = func(this.data[i][j]);
            }
        }
    }

    // Function to add a scalar or matrix to this matrix
    add(value) {
        if (value instanceof Matrix) {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += value.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += value;
                }
            }
        }
    }

    // Function to multiply this matrix with another matrix
    static multiply(matrixA, matrixB) {
        if (matrixA.cols !== matrixB.rows) {
            console.error("Columns of A must match rows of B");
            return undefined;
        }
        let result = new Matrix(matrixA.rows, matrixB.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < matrixA.cols; k++) {
                    sum += matrixA.data[i][k] * matrixB.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Function to subtract one matrix from another
    static subtract(matrixA, matrixB) {
        let result = new Matrix(matrixA.rows, matrixA.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = matrixA.data[i][j] - matrixB.data[i][j];
            }
        }
        return result;
    }

    // Function to transpose the matrix
    static transpose(matrix) {
        let result = new Matrix(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    // Function to create a matrix from an array
    static fromArray(array) {
        let result = new Matrix(array.length, 1);
        for (let i = 0; i < array.length; i++) {
            result.data[i][0] = array[i];
        }
        return result;
    }

    // Function to convert matrix to array
    toArray() {
        let array = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                array.push(this.data[i][j]);
            }
        }
        return array;
    }

    // Function to set all elements to random values between -1 and 1
    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }
}

// Constants
const LEARNING_RATE = 0.1;

// Example usage
const inputNodes = 3;
const hiddenNodes = 3;
const outputNodes = 1;

const neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes);

// Example input and target data
const input = [1, 0, 1];
const target = [0.5];

// Training the neural network
for (let i = 0; i < 10000; i++) {
    neuralNetwork.train(input, target);
}

// Testing the trained network
console.log("Trained Neural Network Output:", neuralNetwork.feedforward(input));

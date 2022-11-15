using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator
{
    public class NeuralNetwork
    {
        int[] m_NetworkShape = {784, 16, 16, 10};

        private int NumOutputs() =>
            m_NetworkShape[^1];
        
        private List<Layer> m_HiddenLayers;

        public NeuralNetwork(int[] networkShape) {
            m_NetworkShape = networkShape;
            m_HiddenLayers = new List<Layer>();
            for (int i = 1; i < m_NetworkShape.Length; i++) {
                int numInputs = m_NetworkShape[i - 1];
                int numNodes = m_NetworkShape[i];
                Layer layer = new Layer(numInputs, numNodes);
                m_HiddenLayers.Add(layer);
            }

            m_HiddenLayers.Last().IsOutputLayer = true;
        }

        public Vector<float> Brain(Vector<float> inputs) {
            if (m_HiddenLayers.Count < 1) return Vector<float>.Build.Dense(0);

            Layer layerRef = m_HiddenLayers[0];
            for (int i = 0; i < m_HiddenLayers.Count; i++) {
                layerRef = m_HiddenLayers[i];
                if (i == 0) {
                    layerRef.ForwardPass(inputs);
                    layerRef.ActivationSigmoid();
                }
                else {
                    Layer lastLayer = m_HiddenLayers[i - 1];
                    layerRef.ForwardPass(lastLayer.nodes);
                    
                    // Don't use activationfunction on output layer
                    if(i != (m_HiddenLayers.Count - 1))
                        layerRef.ActivationSigmoid();
                }
            }

            

            //Console.WriteLine(layerRef.nodes);
            
            return layerRef.nodes;
        }
        
        public Matrix<float> BrainBatch(Matrix<float> inputs) {
            if (m_HiddenLayers.Count < 1) return Matrix<float>.Build.Dense(0, 0);

            Layer layerRef = m_HiddenLayers[0];
            Matrix<float> lastResult = Matrix<float>.Build.Dense(0, 0);
            
            for (int i = 0; i < m_HiddenLayers.Count; i++) {
                layerRef = m_HiddenLayers[i];
                if (i == 0) {
                    lastResult = layerRef.ForwardPassBatch(inputs);
                    layerRef.ActivationReLU();
                }
                else {
                    lastResult = layerRef.ForwardPassBatch(lastResult);
                    
                    // Don't use activationfunction on output layer
                    if(i != (m_HiddenLayers.Count - 1))
                        layerRef.ActivationReLU();
                }
            }

            return lastResult ?? Matrix<float>.Build.Dense(0, 0);
        }

        public float Train(Vector<float> inputs, int label){
            var outPut = Brain(inputs);
            if(outPut == null) return 0.0f;
            
            var probabilityDistOut = SoftMax(outPut);
            Vector<float> labelVec = Vector<float>.Build.Dense(NumOutputs());
            labelVec[label] = 1;
            
            float cost = CostSquare(probabilityDistOut, labelVec);
            //Console.WriteLine(probabilityDistOut);
            //System.Console.WriteLine("Cost: " + cost);

            var delta_o = m_HiddenLayers[^1].nodes - labelVec;

            Vector<float> activationError = Vector<float>.Build.Dense(1);
            for (int i = m_HiddenLayers.Count - 1; i >= 0; i--) {
                var layer = m_HiddenLayers.ElementAt(i);

                if (layer.IsOutputLayer)
                    //activationError = outPut.MapIndexed((index, val) => MathF.Pow(val - labelVec[index], 2));
                    activationError = labelVec - outPut;
                else {
                    var layerAhead = m_HiddenLayers[i + 1];                    
                    activationError = layerAhead.weights.Transpose() * activationError;
                }

                Vector<float> thisLayerInput = Vector<float>.Build.Dense(0);
                if (i == 0)
                    thisLayerInput = inputs;
                else
                    thisLayerInput = m_HiddenLayers[i - 1].nodes;

                if (layer.IsOutputLayer) {
                    layer.weightsNudge +=
                        activationError.PointwiseMultiply(layer.nodes.Map(ActivationSigmoidDerivative))
                            .ToColumnMatrix() * thisLayerInput.ToRowMatrix();
                    
                    layer.biasesNudge +=
                        activationError.PointwiseMultiply(layer.nodes.Map(ActivationFunctionDerivative));
                }
                else {
                    layer.biasesNudge += activationError.PointwiseMultiply(layer.nodes.Map(ActivationFunctionDerivative));
                    layer.weightsNudge += activationError.PointwiseMultiply(layer.nodes.Map(ActivationFunctionDerivative)).ToColumnMatrix() * thisLayerInput.ToRowMatrix();
                }
                
                //Console.WriteLine(activationError);
            }

            
            
            return cost;
        }

        private void BackpropOutput() {
            
        }

        public void UpdateWeightsAndBiases(float learningRate, int batchSize) {
            m_HiddenLayers.ForEach(layer => {
                //Console.WriteLine("Weights:" + layer.weights);
                //Console.WriteLine("Weights Nudge:" +layer.weightsNudge);
                layer.weights += layer.weightsNudge * learningRate * (1 / (float)batchSize);
                layer.biases += layer.biasesNudge * learningRate * (1 / (float)batchSize);
                
                layer.biasesNudge.Clear();
                layer.weightsNudge.Clear();
            });
        }

        public Vector<float> TrainBatch(Matrix<float> inputs, int[] labels) {
            
            var output = BrainBatch(inputs);
            output = SoftMax(output);
            return LossCatergoricalCrossEntropy(output, labels);
        }
        
        public static Vector<float> LossCatergoricalCrossEntropy(Matrix<float> output, int[] labels) {
            Vector<float> lossVec = Vector<float>.Build.Dense(output.RowCount);
            for(int i = 0; i < output.RowCount; i++) {
                var confidence = output.Row(i).At(labels.ElementAt(i));
                var loss = -MathF.Log(confidence, MathF.E);
                lossVec[i] = loss;
            }
            return lossVec;
        }

        public float CostFunctionDerivative(float x) {
            return -(1 / x);
        }

        public static Vector<float> LossSquare(Matrix<float> output, int[] labels) {
            Vector<float> lossVec = Vector<float>.Build.Dense(output.RowCount);
            for(int i = 0; i < output.RowCount; i++) {
                //output.ReduceRows(((floats, vector) => ))
                float sum = 0;
                for (int k = 0; k < output.ColumnCount; k++) {
                    int expVal = labels.ElementAt(i) == k ? 1 : 0; 
                    sum += output[i, k] - expVal;
                }
               
                lossVec[i] = sum;
            }

            return lossVec;
        }


        public static float CostSquare(Vector<float> outputs, Vector<float> targets) {
            float cost = 0.0f;
            for (int i = 0; i < outputs.Count; i++) {
                cost += MathF.Pow((outputs[i] - targets[i]), 2);
            }

            return cost;
        }
        
        // The derivative of the weights * inputs for a node with respect to a specific weight,
        // is the activation value of the previous node of the same index in the last layer

        public static float LossSquareDerivative(float a, float y) {
            return 2 * (a - y);
        }


        public static Vector<float> ActivationFunctionDerivative(Vector<float> vec) {
            return vec.Map(ActivationFunctionDerivative);
        }
        
        public static float ActivationFunctionDerivative(float x) {
            return ActivationSigmoidDerivative(x);
        }
        
        public static float ReluDerivative(float x) {
            return Convert.ToInt32(x > 0);
        }

        public static float ActivationSigmoid(float x) =>
            1 / (1 + MathF.Pow(MathF.E, -x));
        

        public static float ActivationSigmoidDerivative(float x) =>
            ActivationSigmoid(x) * (1 - ActivationSigmoid(x));
        
        
        public void Backprop() {
            for(int i = m_HiddenLayers.Count; i > 0; i--) {
                var weights = m_HiddenLayers.ElementAt(i).weights;
                for(int k = 0; k < weights.ColumnCount; k++){
                    
                }
            }
        }

        public static Vector<float> SoftMax(Vector<float> vec) {
            //var maxValue = vec.Maximum();
            var normalizedValues = vec.Map(val => val / vec.Sum());
            var expValues = normalizedValues.Map(val => {
                var newVal = MathF.Pow(MathF.E, val);
                return newVal;
            });
            var normBase = expValues.Sum();
            return expValues.Map(val => val / normBase);
        }
        
        public static Matrix<float> SoftMax(Matrix<float> mat) {

            for(int i = 0; i < mat.RowCount; i++) {
                // Subtract maximum of each row on each row
                float rowMax = mat.Row(i).Maximum();
                mat.SetRow(i, mat.Row(i) - rowMax);
            }
            //Console.WriteLine(mat);

            var expValues = mat.Map(val => MathF.Pow(MathF.E, val));
           // Console.WriteLine(expValues);
            
            var normBaseVec = expValues.RowSums();
            //Console.WriteLine(normBaseVec);
            // Calculate normalized values for each element in matrix, done per row.
            for(int i = 0; i < expValues.RowCount; i++) {
                for(int k = 0; k < expValues.ColumnCount; k++){
                    expValues[i, k] = expValues[i, k] / normBaseVec[i];
                }
            }
            //Console.WriteLine(expValues);
            //var normalizedValues = expValues.Map(val => val / normBase);
            return expValues;
        }
    }
}
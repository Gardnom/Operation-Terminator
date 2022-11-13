using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator
{
    public class NeuralNetwork
    {
        int[] m_NetworkShape = {2, 4, 4, 3};
        private List<Layer> m_HiddenLayers;

        public NeuralNetwork() {
            m_HiddenLayers = new List<Layer>();
            for (int i = 1; i < m_NetworkShape.Length; i++) {
                int numInputs = m_NetworkShape[i - 1];
                int numNodes = m_NetworkShape[i];
                Layer layer = new Layer(numInputs, numNodes);
                m_HiddenLayers.Add(layer);
            }
        }

        public Vector<float> Brain(Vector<float> inputs) {
            if (m_HiddenLayers.Count < 1) return Vector<float>.Build.Dense(0);

            Layer layerRef = m_HiddenLayers[0];
            for (int i = 0; i < m_HiddenLayers.Count; i++) {
                layerRef = m_HiddenLayers[i];
                if (i == 0) {
                    layerRef.ForwardPass(inputs);
                    layerRef.ActivationReLU();
                }
                else {
                    Layer lastLayer = m_HiddenLayers[i - 1];
                    layerRef.ForwardPass(lastLayer.nodes);
                    
                    // Don't use activationfunction on output layer
                    if(i != (m_HiddenLayers.Count - 1))
                        layerRef.ActivationReLU();
                }
            }

            return layerRef?.nodes ?? Vector<float>.Build.Dense(0);
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
            //System.Console.WriteLine(inputs.ToRowMatrix());
            var asMat = outPut.ToRowMatrix()!;
            var probabilityDistOut = SoftMax(asMat);
            var probabilityDistOutVec = probabilityDistOut.Column(0);
            var costVec = LossCatergoricalCrossEntropy(probabilityDistOut, new int[]{label});
            float cost = costVec.At(0);
            System.Console.WriteLine(cost);
            
            // Backprop
            for(int i = m_HiddenLayers.Count - 1; i >= 0; i --){
                var layer = m_HiddenLayers.ElementAt(i);
                int index = 0;
                for(int k = 0; k < )
                layer.biasesSmudge = layer.biases.Map(x => {
                    return ActivationFunctionDerivate(cost * layer.nodes.At(index));
                });
            }
            
            //var cost = 
            return 0.0f;
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
        
        // The derivative of the weights * inputs for a node with respect to a specific weight,
        // is the activation value of the previous node of the same index in the last layer

        public static float LossSquareDerivative(float a, float y) {
            return 2 * (a - y);
        }

        public static float ActivationFunctionDerivate(float x) {
            return Convert.ToInt32(x > 0);
        } 
        
        public void Backprop() {
            for(int i = m_HiddenLayers.Count; i > 0; i--) {
                var weights = m_HiddenLayers.ElementAt(i).weights;
                for(int k = 0; k < weights.ColumnCount; k++){
                    
                }
            }
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
            Console.WriteLine(expValues);
            //var normalizedValues = expValues.Map(val => val / normBase);
            return expValues;
        }
    }
}
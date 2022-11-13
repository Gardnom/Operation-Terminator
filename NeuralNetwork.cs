using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator
{
    public class NeuralNetwork
    {
        int[] m_NetworkShape = {1, 4, 4, 3};
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

        public Vector<float> TrainBatch(Matrix<float> inputs, int[] labels) {
            Vector<float> lossVec = Vector<float>.Build.Dense(inputs.RowCount);
            var output = BrainBatch(inputs);
            output = SoftMax(output);
            for(int i = 0; i < output.RowCount; i++) {
                var confidence = output.Row(i).At(labels.ElementAt(i));
                var loss = -MathF.Log(confidence, MathF.E);
                lossVec[i] = loss;
            }
            return lossVec;
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
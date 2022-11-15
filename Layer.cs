using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator
{
    public class Layer
    {
        public int m_NNodes, m_NInputs;
        public Vector<float> nodes;
        public Vector<float> biases;
        public Matrix<float> weights;

        public Vector<float> valuesPreActivation;
        //public Vector<float> inputs;

        public Vector<float> desiredValues;
        public Vector<float> biasesNudge;
        public Matrix<float> weightsNudge;

        public bool IsOutputLayer = false;

        private System.Random _random = new Random();

        public Layer(int nInputs, int nNodes) {
            m_NInputs = nInputs;
            m_NNodes = nNodes;

            nodes = Vector<float>.Build.Dense(nNodes);
            biases = Vector<float>.Build.Random(nNodes);
            weights = Matrix<float>.Build.Random(nNodes, nInputs);
            valuesPreActivation = Vector<float>.Build.Dense(nNodes);

            
            desiredValues = Vector<float>.Build.Dense(nNodes);
            biasesNudge = Vector<float>.Build.Dense(nNodes);
            weightsNudge = Matrix<float>.Build.Random(nNodes, nInputs);
        }

        public void ForwardPass(Vector<float> inputs) {
            if (inputs.Count != m_NInputs) {
                Console.WriteLine("Wrong number of inputs in forward pass");
                return;
            }
            nodes = weights * inputs + biases;
            nodes.CopyTo(valuesPreActivation);
        }
        
        public Matrix<float> ForwardPassBatch(Matrix<float> inputs) {
            return inputs * weights;
        }

        public void UpdateWeights(float learningRate) {
            weights -= weightsNudge * learningRate;
            /*for (int i = 0; i < m_NNodes; i++) {
                for (int k = 0; k < m_NInputs; k++) {
                    
                }
            }*/
        }

        public void ActivationSigmoid() {
            nodes.MapInplace(val => 1 / (1 + MathF.Pow(MathF.E, -val)));
        }
        public void ActivationReLU() {
            nodes.MapInplace(val => {
                if (val < 0) return 0;
                return val;
            });
        }
    }
}
# Toy Neural Network (NumPy Only)

As part of my autonomous exploration of modern technologies, I wanted to understand how Artificial Neural Networks actually work **"under the hood"**, rather than just calling an API. 

This project is a minimalist, from-scratch implementation of a 2-layer Neural Network using only pure Python and the `NumPy` library. 

### What it does
The network is trained to solve the classic **XOR logic problem**, which is impossible for a simple linear classifier to solve. It uses:
- The `Sigmoid` activation function
- Feedforward propagation
- Backpropagation (calculating derivatives to update weights)

### AI Tutoring Workflow
*I used ChatGPT not to write the code for me, but as an **interactive tutor**. I asked it to explain the calculus behind backpropagation and how to translate partial derivatives into NumPy matrix multiplications. This hands-on project helped bridge the gap between my discrete mathematics background from RWTH Aachen and modern machine learning concepts.*

### How to run
```bash
pip install numpy
python simple_nn.py

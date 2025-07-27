# Physics-Informed Neural Network (PINN) for 2D Steady-State Heat Transfer with and without Generalization

## 🚀 Project Overview

This repository explores the fascinating yet challenging frontier of using Physics-Informed Neural Networks (PINNs) to solve the 2D steady-state heat equation (Laplace equation, $\nabla^2 T = 0$) with a unique focus: **generalization to unseen boundary conditions**. Unlike conventional PINN applications that learn a single, fixed PDE solution, this project aimed to develop a "universal solver" capable of inferring temperature fields across a domain given *any* combination of boundary temperatures, without requiring retraining.

The journey highlighted significant complexities inherent in training generalized PINNs, from managing high-dimensional input spaces to overcoming persistent training stability issues like the "averaging failure mode." This README details the architectural choices, the encountered challenges, and the insights gained into the practical limitations of current PINN generalization strategies.

## ✨ Features & Approaches Explored

* **Generalized PINN Architecture:** The network input was augmented to include not only spatial coordinates ($X, Y$) but also the four boundary temperatures ($T_{\text{top}}, T_{\text{left}}, T_{\text{bottom}}, T_{\text{right}}$), transforming the model into a meta-learner.
* **Two-Stage Optimization Strategy:** Employed a sequential training approach using:
    * **AdamW (Stage 1):** For rapid initial convergence and effective weight regularization to combat overfitting.
    * **Adam with Low Learning Rate (Stage 2):** For fine-tuning and precise optimization.
* **Custom PINN Implementation:** Developed a robust, mini-batch capable PINN structure from scratch using TensorFlow/Keras, featuring custom loss functions for physics and boundary conditions.
* **SciANN Integration:** Re-implemented the model using SciANN, a high-level library for scientific deep learning, to validate findings and confirm that challenges were problem-inherent, not code-specific.
* **Activation Function Exploration:** Tested various activation functions for their impact on generalization, including:
    * `sin`
    * `tanh`
    * `l-tanh` (learnable hyperbolic tangent)
    * `swish`
* **Data Scaling:** Investigated the impact of data normalization/standardization on model performance.

## 🚧 Challenges & Learnings

Developing a truly generalized PINN proved to be an uphill battle, revealing several critical challenges:

* **High-Dimensional Input Space and Model Complexity:** Encoding diverse boundary conditions dramatically increased input dimensionality, making the learning task exponentially more complex for the neural network. Furthermore, for the network to capture such complex, multi-faceted relationships, it often needed to be **extremely large**, with a substantial number of layers and neurons. This increased model complexity, in turn, escalated the computational burden.
* **Discontinuity & Sharp Gradients:** PINNs struggled to simultaneously satisfy boundary conditions (which often introduce sharp changes) and maintain solution smoothness in the interior.
* **Lack of Inductive Bias for Generalization:** Unlike learning a single function, learning a "function-to-function" mapping from BCs to solution fields is a far more constrained problem for standard neural networks.
* **Data Requirements & Generation:** The need for a vast, highly diverse dataset of BCs and corresponding solutions (from FDM) was computationally demanding and challenging to generate comprehensively. Even with available data, the training process for such large models and complex generalization tasks was exceptionally time-consuming, often **exceeding 19 hours to complete a single run**, requiring significant computational resources and patience.
* **"Averaging Failure Mode":** A significant and recurring challenge encountered was the network's tendency to fall into an "averaging failure mode." In this scenario, instead of learning the complex spatial temperature distribution governed by the specific boundary conditions, the model would often predict a nearly constant temperature value across the entire domain, effectively approximating the average of the boundary conditions or the training data's output values.


## 🛠️ Setup & Installation

To get this project up and running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/manisalehi/2D-heat-equation-pinn](https://github.com/manisalehi/2D-heat-equation-pinn)
    cd 2D-heat-equation
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The `requirements.txt` file specifies all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    **Important Note on TensorFlow/SciANN Compatibility:**
    For experiments involving SciANN, it's crucial to be aware of its compatibility with specific TensorFlow versions, as SciANN is no longer actively maintained. This project was tested with **TensorFlow 2.12.0**, though TensorFlow 2.10.0 is also often recommended for SciANN. The `requirements.txt` is set to ensure a compatible environment.

## 🚀 How to Run

This project's workflow is primarily demonstrated within the Jupyter Notebook.

1.  **Ensure File Presence:** After cloning the repository, make sure the following files are in the same directory as `PINN.ipynb`:
    * `pinnTools.py`
    * `FDM.py`
2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
3.  **Execute Cells in `PINN.ipynb`:** Open `PINN.ipynb` in your browser and execute its cells sequentially. This notebook will guide you through:
    * Data generation (using `FDM.py`)
    * Setting up the PINN model (using `pinnTools.py` and custom code)
    * Training the model (including the two-stage optimization)
    * Evaluating its performance for both fixed and generalized boundary conditions
    * Demonstrating the challenges encountered.

## 📊 Results & Discussion

Crucially, when trained for a **fixed set of boundary conditions**, the underlying PINN solver model developed in this project demonstrated **absolute effectiveness and excellent accuracy** in solving the 2D steady-state heat equation. This success was largely attributable to the meticulous design of the custom training loop, which allowed for precise control over loss components and optimization strategies.

However, the primary, more ambitious goal of achieving **robust generalization** to *unseen* boundary conditions proved to be a formidable challenge. As thoroughly demonstrated through extensive experimentation within this project, robust generalization cannot be reliably achieved with small to moderately sized models, even those as large as having **20 hidden layers with 128 units each**. The persistent "averaging failure mode" and the sheer complexity of the high-dimensional input space proved to be significant hurdles.

This project serves as an **excellent introductory resource for individuals new to Physics-Informed Neural Networks**. It not only provides a working solver for a fundamental PDE but also transparently highlights some of the most common and challenging pitfalls in PINN research, particularly concerning generalization. By exploring the code and the documented challenges, newcomers can gain valuable insights into the practical complexities of training PINNs and learn strategies (and their limitations) for attempting to overcome them.

## 📚 References

A selection of key references that informed this project's understanding and implementation of PINNs and their generalization capabilities:

* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations. *arXiv preprint arXiv:1711.10561*.
* Lu, D., Lu, L., & Karniadakis, G. E. (2021). DeepONet: Learning nonlinear operators for predicting solution manifolds of PDEs. *Nature Machine Intelligence*, 3(3), 248-259.
* Yang, K., Chen, S., H. Meng, M. Zheng, & Li, B. (2020). Learning to solve partial differential equations with physics-informed neural networks for inverse problems. *Journal of Computational Physics*, 423, 109849.
* Wang, L., Ma, T., Chen, B., & Liu, Y. (2021). A Meta-Learning Framework for Physics-Informed Neural Networks. In *International Conference on Learning Representations (ICLR)*.
* Zheng, M., Li, S., & Luo, L. (2021). Physics-informed neural networks for modeling and solving partial differential equations with varying parameters. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 31(12), 123102.
* Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). Locally adaptive activation functions with slope recovery term for deep and physics-informed neural networks. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 476(2242), 20200334. (arXiv:1909.12228)
* Haghighat, E., & Juanes, R. (2021). SciANN: A Keras/TensorFlow wrapper for scientific computations and physics-informed deep learning using artificial neural networks. *Computer Methods in Applied Mechanics and Engineering*, 373, 113552. (arXiv:2005.08803, GitHub: [https://github.com/sciann/sciann](https://github.com/sciann/sciann))
* Kim, H. K., Park, J. H., & Kim, D. (2021). HyperNetworks: A Tutorial and Survey. *Journal of Machine Learning Research*, 22(21), 1-62.
* McClenny, L., & Braga-Neto, J. (2020). How to avoid trivial solutions in physics-informed neural networks. *arXiv preprint arXiv:2010.00764*.
* Wang, Y., Lu, L., & Karniadakis, G. E. (2021). Physics-informed Neural Networks with Periodic Activation Functions for Solute Transport in Heterogeneous Porous Media. *Journal of Computational Physics*, 446, 110651.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/manisalehi/2D-heat-equation/issues).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Seyed Mani Seyed SalehiZadeh – manisalehi2004@gmail.com

Project Link: [https://github.com/manisalehi/2D-heat-equation-pinn](https://github.com/manisalehi/2D-heat-equation-pinn)

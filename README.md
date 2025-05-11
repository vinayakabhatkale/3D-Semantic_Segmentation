
<p align="center">
  <img width="40%" height="40%" src="./illustrations/teaser.png">
</p>


3D Workspace Exploration
This repository focuses on 3D semantic segmentation of a robotic workspace using point cloud data. The goal is to segment and understand various components in a 3D environment to facilitate robotic perception and interaction.

⚠️ For generating 3D synthetic training data, please refer to the companion repository 3D Synthetic Data Generator (coming soon).

📦 Docker Setup
To ensure a reproducible environment, this project is containerized using Docker.

1. Build the Docker container
bash
Copy
Edit
docker build -t workspace_container .
2. Start the container
Use the provided script to start the container:

bash
Copy
Edit
./start.sh
This script handles volume mounting and launches an interactive shell into the container.

3. Restart an existing container
If you exited the container and want to re-enter it:

bash
Copy
Edit
./restart.sh
🏋️ Training
Once inside the container, you can run the training script:

bash
Copy
Edit
python train.py
Output
Training logs, model checkpoints, and visualizations are saved in the output/ directory.

📊 Evaluation
To evaluate a trained model:

bash
Copy
Edit
python eval.py
Results will be stored in the output/ folder as well.

📂 Directory Structure
php
Copy
Edit
.
├── Dockerfile
├── start.sh
├── restart.sh
├── train.py
├── eval.py
├── output/               # Logs, checkpoints, visualizations
├── data/                 # Place training data here
└── README.md
🧪 Training Data
Training data should be organized as point clouds (e.g., .npy, .ply, .pcd). For dataset generation in simulation, see:

🔗 3D Synthetic Data Generator (Coming Soon)



The repo is based on Deepviewagg and used as a 3D robotic workspace exploration using semantic segmentation.
 @article{robert2022dva,
  title={Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation},
  author={Robert, Damien and Vallet, Bruno and Landrieu, Loic},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}







📃 License
This project is released under the MIT License.

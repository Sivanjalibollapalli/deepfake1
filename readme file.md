🔍 DeepFake Detection Framework Using Modified ResNet and NPR Techniques
📌 Abstract
Deepfakes and synthetic media have rapidly emerged as a significant threat in the digital age. These highly convincing manipulated visuals pose serious risks to personal privacy, national security, and the integrity of information. This project presents a novel deepfake detection system that addresses limitations in current detection frameworks by:

Rethinking up-sampling operations in CNNs.

Introducing Non-Photorealistic Rendering (NPR) to highlight subtle artifacts.

Utilizing a modified ResNet-based architecture for robust feature extraction.

Achieving detection accuracy of 92%+ across various deepfake generation techniques including ProGAN, StyleGAN, and BigGAN.

A web-based interface further enables real-time image analysis, making the system both accessible and practical for deployment.

🧠 Key Features
✅ High-Accuracy Detection – 92%+ detection rate on multiple manipulation techniques.

🖼️ Non-Photorealistic Rendering (NPR) – Enhances visual inconsistencies in fake images.

🔗 Modified ResNet Architecture – Optimized for cross-domain deepfake detection.

🌐 Web Interface – Easy-to-use GUI for real-time detection.

🔁 Cross-Model Generalization – Trained on outputs from multiple deepfake generators.

🛠️ Project Structure
php
Copy
Edit
deepfake-detector/
│
├── models/                  # Trained model weights and ResNet modifications
├── npr_filters/            # Non-photorealistic rendering filters
├── dataset/                # Training and test image datasets
├── webapp/                 # Web interface for real-time detection
│   ├── static/
│   ├── templates/
│   └── app.py
├── utils/                  # Preprocessing, augmentation, and analysis tools
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
🚀 Getting Started
Prerequisites
Python 3.8+

pip

Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/deepfake-detector-npr.git
cd deepfake-detector-npr
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the web application

bash
Copy
Edit
cd webapp
python app.py
Access the detector Open your browser and go to http://localhost:5000

🔬 Methodology
CNN Rethinking: Traditional upsampling in GAN-generated images introduces specific statistical patterns. Our system re-evaluates these operations for better artifact detection.

NPR Techniques: Artistic filters (e.g., edge maps, sketch effects) reveal inconsistencies that are visually suppressed in realistic deepfakes.

Training Dataset: Includes a balanced mix of real images and synthetic images generated by ProGAN, StyleGAN, and BigGAN.

📈 Results
Model Type	Accuracy	Precision	Recall
ProGAN	94.3%	93.1%	92.5%
StyleGAN	92.7%	91.8%	90.4%
BigGAN	91.5%	90.2%	89.8%
Average	92.8%	91.7%	90.9%
💡 Future Work
Expand detection capabilities to video-level deepfakes.

Explore lightweight models for edge and mobile deployment.

Incorporate adversarial training for robustness.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
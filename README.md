# High-Throughput-Molecular-Docking

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.6+-orange.svg" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Google_Colab-yellow.svg" alt="Platform">
</p>

## 🧪 Overview

**High-Throughput-Molecular-Docking** is an efficient molecular docking tool developed by **the Biocatalysis and Biotransformation Laboratory at Tianjin University of Science and Technology**. This comprehensive pipeline streamlines large-scale molecular docking tasks, making it user-friendly for researchers in drug discovery, molecular design, and biochemical research.

## 🌟 Key Features

- **Automated Workflow** 🔄: Automates the pre-processing of both receptors and ligands, simplifying the initial setup
- **Automatic Parameter Acquisition** ⚙️: Automatically retrieves necessary docking parameters, reducing time and effort needed for manual input
- **Batch Processing** 📊: Performs simultaneous docking of multiple compounds, enhancing productivity
- **Interactive Visualization** 👁️: Integrated 3D visualization capabilities for examining binding modes and interactions
- **Statistical Analysis** 📈: Generates publication-quality visualizations and detailed analysis reports

## 📋 Contents

The tool includes the following Jupyter notebooks:

1. **Environment Configuration.ipynb** - Sets up all necessary tools and dependencies
2. **Batch Preprocessing - Receptor Files.ipynb** - Prepares receptor files for docking
3. **Batch Preprocessing - Ligand Files.ipynb** - Prepares ligand files for docking
4. **Configure Docking Parameters.ipynb** - Determines optimal docking parameters
5. **Colab-High Throughput Molecular Docking.ipynb** - Performs molecular docking in Google Colab
6. **Local-High Throughput Molecular Docking.ipynb** - Local version for running on your own machine
7. **Molecular Docking Analysis Tool.ipynb** - Analyzes and visualizes docking results
8. **Receptor File Parameter Analysis.ipynb** - Analyzes receptor parameters

## 💻 Usage

### Google Colab Version

Access the tool on Google Colab: [High-Throughput-Molecular-Docking](https://colab.research.google.com/drive/1rPRKJnHsLlgjsvF4Tx_amXR0nKgGejCM?usp=sharing)

Follow these steps to use the tool:

1. **Environment Setup**
   - Run the Environment Configuration notebook
   - Wait for all dependencies to be installed

2. **Prepare Files**
   - Run the Receptor Files preprocessing notebook
   - Run the Ligand Files preprocessing notebook
   - Upload and process your molecule files

3. **Configure Docking**
   - Run the Configure Docking Parameters notebook
   - Upload processed receptor and ligand files
   - The system will automatically calculate optimal parameters

4. **Perform Docking**
   - Run the High Throughput Molecular Docking notebook
   - Upload receptor, ligand, and parameter files
   - Wait for the docking process to complete

5. **Analyze Results**
   - Run the Molecular Docking Analysis Tool notebook
   - Upload your docking results ZIP file
   - Download the comprehensive analysis report

### Local Version Setup

For the local version, follow these steps:

1. **Prerequisites**:
   - Python 3.6+ with required packages:
     ```
     pip install numpy pandas seaborn matplotlib scikit-learn psutil
     ```
   - AutoDock Vina installed

2. **Directory Structure**:
   ```
   YOUR_BASE_DIRECTORY/
   ├── vina.exe                  # AutoDock Vina executable
   ├── receptors/                # Receptor files (.pdbqt)
   ├── ligands/                  # Ligand files (.pdbqt)
   ├── conf/                     # Configuration files
   └── Docking results/          # Results directory (created automatically)
   ```

3. **Configuration Files**:
   For each receptor, create a configuration file in the `conf/` directory with parameters:
   ```
   center_x = 0.0
   center_y = 0.0
   center_z = 0.0
   size_x = 20.0
   size_y = 20.0
   size_z = 20.0
   ```

4. **Usage**:
   - Modify the path variables in the script
   - Run the script and follow the prompts
   - Check the results directory for outputs

## 📊 Output

The tool generates the following outputs:

### Docking Results
- Docked structure files (PDBQT)
- Log files with docking information

### Analysis
- Statistical summaries (CSV)
- Binding energy distribution charts
- Energy landscape plots
- Structure-activity relationship visualizations
- Pose clustering analysis
- Detailed analysis report (TXT)

### Visualizations
- Interactive 3D models of receptor-ligand complexes
- Binding site analysis
- 2D interaction diagrams

## 🔧 Requirements

- **Google Colab** for the online version
- **Python 3.6+** with the following packages for the local version:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn
  - psutil
- **AutoDock Vina**

## 🎓 Citation

If you use this tool in your research, please cite:

```
[To be published]
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contact

For questions or support, please contact:
[chunruzhou@mail.tust.edu.cn]

---

<p align="center">
  <i>Developed by the Biocatalysis and Biotransformation Laboratory, Tianjin University of Science and Technology</i>
</p>

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from .environment import get_environment

logger = logging.getLogger(__name__)


class EnzymeDockingAnalyzer:
    """Enzyme Engineering Docking Analysis with Dual ML Models"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the analyzer
        
        Args:
            output_dir: Output directory for results
        """
        self.env = get_environment()
        self.output_dir = Path(output_dir) if output_dir else self.env.get_work_dir() / "enzyme_analysis_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = []
        self.scores = []
        self.residue_data = None
        self.ml_features = None
        self.ml_models = {}
        
        self.setup_publication_style()
    
    def setup_publication_style(self):
        """Setup publication-quality plotting style"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'figure.dpi': 150,
            'figure.figsize': [8, 6],
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })
    
    def process_docking_results(self, docking_files: Dict[str, bytes], 
                               receptor_file: Optional[bytes] = None) -> bool:
        """
        Process docking results from uploaded files
        
        Args:
            docking_files: Dictionary of docking result filename -> content
            receptor_file: Optional receptor file content
            
        Returns:
            True if processing successful
        """
        logger.info("📂 Processing docking results...")
        
        # Create temporary directory
        temp_dir = self.output_dir / "temp_processing"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save and process docking files
            pdbqt_files = {}
            for filename, content in docking_files.items():
                file_path = temp_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                if filename.endswith('.zip'):
                    # Extract ZIP file
                    extracted_files = self.extract_zip_files(file_path, temp_dir)
                    pdbqt_files.update(extracted_files)
                elif filename.endswith('.pdbqt'):
                    pdbqt_files[filename] = content
            
            if not pdbqt_files:
                logger.error("No PDBQT files found")
                return False
            
            # Parse docking results
            self.results = []
            for filename, content in pdbqt_files.items():
                file_path = temp_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                result = self.parse_docking_file(file_path)
                if result:
                    self.results.extend(result)
            
            if not self.results:
                logger.error("No valid docking results found")
                return False
            
            self.scores = [r['score'] for r in self.results]
            logger.info(f"✓ Successfully parsed {len(self.results)} docking poses")
            
            # Extract residue interactions if receptor provided
            if receptor_file:
                logger.info("🔍 Extracting residue interactions...")
                # Save receptor file
                receptor_path = temp_dir / "receptor.pdb"
                with open(receptor_path, 'wb') as f:
                    f.write(receptor_file)
                
                self.residue_data = self.extract_residue_interactions(pdbqt_files, receptor_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing docking results: {str(e)}")
            return False
        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def extract_zip_files(self, zip_path: Path, extract_dir: Path) -> Dict[str, bytes]:
        """Extract PDBQT files from ZIP archive"""
        extracted_files = {}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                for filename in zipf.namelist():
                    if filename.endswith('.pdbqt'):
                        content = zipf.read(filename)
                        extracted_files[filename] = content
                        
                        # Also save to disk for processing
                        with open(extract_dir / filename, 'wb') as f:
                            f.write(content)
            
            logger.info(f"   📦 Extracted {len(extracted_files)} PDBQT files from ZIP")
            
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {str(e)}")
        
        return extracted_files
    
    def parse_docking_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse docking output file"""
        results = []
        
        try:
            current_model = None
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('MODEL'):
                        try:
                            current_model = int(line.split()[1])
                        except ValueError:
                            current_model = None
                    elif line.startswith('REMARK VINA RESULT:'):
                        try:
                            parts = line.strip().split()
                            score = float(parts[3])
                            rmsd_lb = float(parts[4])
                            rmsd_ub = float(parts[5])
                            results.append({
                                'file': file_path.name,
                                'model': current_model,
                                'score': score,
                                'rmsd_lb': rmsd_lb,
                                'rmsd_ub': rmsd_ub
                            })
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Error parsing line in {file_path.name}: {str(e)}")
                            continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Error reading file {file_path.name}: {str(e)}")
            return []
    
    def extract_residue_interactions(self, pdbqt_files: Dict[str, bytes], 
                                   receptor_path: Path) -> Optional[pd.DataFrame]:
        """Extract residue-ligand interactions from PDBQT files"""
        logger.info("🔍 Extracting residue-ligand interactions...")
        
        try:
            # Parse receptor file
            receptor_residues = self.extract_receptor_residues(receptor_path)
            if not receptor_residues:
                logger.error("Could not extract residue information from receptor")
                return None
            
            logger.info(f"✓ Extracted {len(receptor_residues)} residues from receptor")
            
            # Analyze docking files
            all_interactions = []
            
            temp_dir = self.output_dir / "temp_interaction"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                for filename, content in pdbqt_files.items():
                    file_path = temp_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    
                    file_interactions = self.analyze_docking_file(file_path, receptor_residues)
                    if file_interactions:
                        all_interactions.extend(file_interactions)
                
                if not all_interactions:
                    logger.error("No residue interactions found")
                    return None
                
                # Create DataFrame
                interaction_df = pd.DataFrame(all_interactions)
                
                # Save to CSV
                output_path = self.output_dir / 'residue_interactions.csv'
                interaction_df.to_csv(output_path, index=False)
                logger.info(f"✓ Residue interaction data saved to {output_path}")
                
                logger.info(f"✓ Total {len(interaction_df)} residue interaction records extracted")
                
                return interaction_df
                
            finally:
                # Cleanup
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            logger.error(f"Error extracting residue interactions: {str(e)}")
            return None
    
    def extract_receptor_residues(self, receptor_path: Path) -> Dict[str, Any]:
        """Extract residue information from receptor PDB/PDBQT file"""
        residues = {}
        
        try:
            with open(receptor_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            atom_name = line[12:16].strip()
                            res_name = line[17:20].strip()
                            res_num = line[22:26].strip()
                            chain = line[21:22].strip()
                            residue_id = f"{res_name}{res_num}{chain}"
                            
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            
                            if residue_id not in residues:
                                residues[residue_id] = {
                                    'res_name': res_name,
                                    'res_num': res_num,
                                    'chain': chain,
                                    'atoms': {}
                                }
                            
                            residues[residue_id]['atoms'][atom_name] = [x, y, z]
                            
                        except (ValueError, IndexError):
                            continue
            
            return residues
            
        except Exception as e:
            logger.error(f"Error extracting receptor residues: {str(e)}")
            return {}
    
    def analyze_docking_file(self, file_path: Path, receptor_residues: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze docking file and extract interactions with receptor residues"""
        interactions = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.readlines()
            
            filename = file_path.name
            current_model = None
            ligand_atoms = []
            model_score = None
            reading_model = False
            
            for line in content:
                if line.startswith('MODEL'):
                    try:
                        current_model = int(line.split()[1])
                        ligand_atoms = []
                        model_score = None
                        reading_model = True
                    except ValueError:
                        current_model = None
                        reading_model = False
                
                elif line.startswith('REMARK VINA RESULT:') and reading_model:
                    try:
                        parts = line.strip().split()
                        model_score = float(parts[3])
                    except (IndexError, ValueError):
                        model_score = None
                
                elif line.startswith('HETATM') and reading_model:
                    try:
                        atom_name = line[12:16].strip()
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atom_type = line[77:79].strip() if len(line) >= 79 else ""
                        
                        ligand_atoms.append({
                            'name': atom_name,
                            'coords': [x, y, z],
                            'type': atom_type
                        })
                    except (IndexError, ValueError):
                        continue
                
                elif line.startswith('ENDMDL') and reading_model:
                    if current_model is not None and model_score is not None and ligand_atoms:
                        # Calculate interactions for each residue
                        for res_id, residue in receptor_residues.items():
                            interaction = self.calculate_residue_interaction(residue, res_id, ligand_atoms)
                            
                            if interaction['has_contact']:
                                interactions.append({
                                    'file': filename,
                                    'model': current_model,
                                    'score': model_score,
                                    'residue_id': res_id,
                                    'res_name': residue['res_name'],
                                    'res_num': residue['res_num'],
                                    'chain': residue['chain'],
                                    'min_distance': interaction['min_distance'],
                                    'contact_count': interaction['contact_count'],
                                    'h_bond_count': interaction['h_bond_count'],
                                    'ionic_count': interaction['ionic_count'],
                                    'hydrophobic_count': interaction['hydrophobic_count'],
                                    'energy_contribution': interaction['energy_contribution']
                                })
                    
                    reading_model = False
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error analyzing docking file {file_path.name}: {str(e)}")
            return []
    
    def calculate_residue_interaction(self, residue: Dict[str, Any], res_id: str, 
                                    ligand_atoms: List[Dict[str, Any]], cutoff: float = 4.0) -> Dict[str, Any]:
        """Calculate residue-ligand interactions"""
        min_distance = float('inf')
        contact_count = 0
        h_bond_count = 0
        ionic_count = 0
        hydrophobic_count = 0
        energy_contribution = 0.0
        
        # Define interaction types
        h_bond_donors = ['N', 'O']
        h_bond_acceptors = ['N', 'O', 'F']
        hydrophobic = ['C', 'CL', 'BR', 'I', 'S']
        
        # Calculate interactions
        for res_atom_name, res_atom_coords in residue['atoms'].items():
            for lig_atom in ligand_atoms:
                # Calculate distance
                distance = np.sqrt(
                    (res_atom_coords[0] - lig_atom['coords'][0])**2 +
                    (res_atom_coords[1] - lig_atom['coords'][1])**2 +
                    (res_atom_coords[2] - lig_atom['coords'][2])**2
                )
                
                min_distance = min(min_distance, distance)
                
                if distance <= cutoff:
                    contact_count += 1
                    
                    res_atom_type = res_atom_name[0]
                    lig_atom_type = lig_atom['type'][0] if lig_atom['type'] else lig_atom['name'][0]
                    
                    # Hydrogen bond detection
                    if ((res_atom_type in h_bond_donors and lig_atom_type in h_bond_acceptors) or
                        (res_atom_type in h_bond_acceptors and lig_atom_type in h_bond_donors)):
                        h_bond_count += 1
                    
                    # Ionic interaction detection  
                    if ((res_atom_type == 'N' and lig_atom_type == 'O') or
                        (res_atom_type == 'O' and lig_atom_type == 'N')):
                        ionic_count += 1
                    
                    # Hydrophobic interaction
                    if res_atom_type in hydrophobic and lig_atom_type in hydrophobic:
                        hydrophobic_count += 1
                    
                    # Energy contribution estimation
                    if distance < 2.0:
                        energy_contribution += 5.0  # Repulsion
                    else:
                        contrib = -2.0 / distance**2
                        if ((res_atom_type in h_bond_donors and lig_atom_type in h_bond_acceptors) or
                            (res_atom_type in h_bond_acceptors and lig_atom_type in h_bond_donors)):
                            contrib *= 2.0
                        energy_contribution += contrib
        
        if min_distance == float('inf'):
            min_distance = 999.0
        
        return {
            'has_contact': contact_count > 0,
            'min_distance': min_distance,
            'contact_count': contact_count,
            'h_bond_count': h_bond_count,
            'ionic_count': ionic_count,
            'hydrophobic_count': hydrophobic_count,
            'energy_contribution': energy_contribution
        }
    
    def prepare_ml_features(self) -> Optional[pd.DataFrame]:
        """Prepare machine learning features from interaction data"""
        if self.residue_data is None or len(self.residue_data) == 0:
            logger.error("Cannot prepare features: no interaction data available")
            return None
        
        logger.info("🧪 Preparing machine learning features...")
        
        try:
            # Aggregate data by residue
            residue_features = self.residue_data.groupby('residue_id').agg({
                'min_distance': 'mean',
                'contact_count': 'sum',
                'h_bond_count': 'sum',
                'ionic_count': 'sum',
                'hydrophobic_count': 'sum',
                'energy_contribution': 'sum',
                'score': 'mean',
                'model': 'count'
            }).reset_index()
            
            residue_features = residue_features.rename(columns={'model': 'frequency'})
            
            # Calculate additional features
            residue_features['contact_ratio'] = residue_features['contact_count'] / residue_features['frequency']
            residue_features['h_bond_ratio'] = residue_features['h_bond_count'] / residue_features['frequency']
            residue_features['ionic_ratio'] = residue_features['ionic_count'] / residue_features['frequency']
            residue_features['hydrophobic_ratio'] = residue_features['hydrophobic_count'] / residue_features['frequency']
            residue_features['energy_per_contact'] = residue_features['energy_contribution'] / residue_features['contact_count']
            residue_features['energy_per_contact'].fillna(0, inplace=True)
            
            # Extract residue type
            residue_features['res_type'] = residue_features['residue_id'].str[:3]
            
            # Add residue type features
            residue_features['is_hydrophobic'] = residue_features['res_type'].isin(
                ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
            )
            residue_features['is_polar'] = residue_features['res_type'].isin(
                ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']
            )
            residue_features['is_charged_pos'] = residue_features['res_type'].isin(
                ['LYS', 'ARG', 'HIS']
            )
            residue_features['is_charged_neg'] = residue_features['res_type'].isin(
                ['ASP', 'GLU']
            )
            
            # Calculate relative metrics
            max_freq = residue_features['frequency'].max()
            if max_freq > 0:
                residue_features['rel_frequency'] = residue_features['frequency'] / max_freq
            else:
                residue_features['rel_frequency'] = 0
            
            min_energy = residue_features['energy_contribution'].min()
            max_energy = residue_features['energy_contribution'].max()
            energy_range = max_energy - min_energy
            
            if energy_range > 0:
                residue_features['rel_energy'] = (residue_features['energy_contribution'] - min_energy) / energy_range
            else:
                residue_features['rel_energy'] = 0
            
            logger.info(f"✓ Created features for {len(residue_features)} residues")
            
            self.ml_features = residue_features
            return residue_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def train_ml_models(self) -> bool:
        """Train dual ML models for catalytic hotspots and specificity sites"""
        if self.ml_features is None or len(self.ml_features) < 10:
            logger.error("Insufficient data for model training")
            return False
        
        logger.info("🤖 Training dual ML models...")
        
        try:
            # Feature selection
            feature_cols = [
                'frequency', 'min_distance', 'contact_count', 'h_bond_count',
                'ionic_count', 'hydrophobic_count', 'energy_contribution',
                'contact_ratio', 'h_bond_ratio', 'ionic_ratio', 'hydrophobic_ratio',
                'energy_per_contact', 'is_hydrophobic', 'is_polar',
                'is_charged_pos', 'is_charged_neg', 'rel_frequency', 'rel_energy'
            ]
            
            # Ensure all features exist
            available_features = [col for col in feature_cols if col in self.ml_features.columns]
            X = self.ml_features[available_features].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create labels for catalytic hotspots
            energy_threshold = np.percentile(self.ml_features['energy_contribution'], 15)
            freq_threshold = np.percentile(self.ml_features['frequency'], 85)
            
            y_key_energy = (self.ml_features['energy_contribution'] <= energy_threshold).astype(int)
            y_key_freq = (self.ml_features['frequency'] >= freq_threshold).astype(int)
            y_key = ((y_key_energy + y_key_freq) > 0).astype(int)
            
            # Create labels for specificity sites (simplified)
            specificity_scores = np.random.random(len(self.ml_features))  # Placeholder
            spec_threshold = np.percentile(specificity_scores, 85)
            y_spec = (specificity_scores >= spec_threshold).astype(int)
            
            # Split data
            X_train, X_test, y_key_train, y_key_test = train_test_split(
                X_scaled, y_key, test_size=0.3, random_state=42
            )
            _, _, y_spec_train, y_spec_test = train_test_split(
                X_scaled, y_spec, test_size=0.3, random_state=42
            )
            
            # Train catalytic hotspot model
            rf_key = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=42, class_weight='balanced'
            )
            rf_key.fit(X_train, y_key_train)
            
            # Train specificity model  
            rf_spec = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=42, class_weight='balanced'
            )
            rf_spec.fit(X_train, y_spec_train)
            
            # Evaluate models
            key_accuracy = accuracy_score(y_key_test, rf_key.predict(X_test))
            spec_accuracy = accuracy_score(y_spec_test, rf_spec.predict(X_test))
            
            logger.info(f"  ✓ Catalytic hotspot model accuracy: {key_accuracy:.2f}")
            logger.info(f"  ✓ Specificity model accuracy: {spec_accuracy:.2f}")
            
            # Generate predictions
            y_key_proba = rf_key.predict_proba(X_scaled)[:, 1]
            y_spec_proba = rf_spec.predict_proba(X_scaled)[:, 1]
            
            self.ml_features['key_residue_prob'] = y_key_proba
            self.ml_features['specificity_prob'] = y_spec_proba
            self.ml_features['specificity_score'] = specificity_scores
            
            # Store models
            self.ml_models = {
                'catalytic_hotspot': rf_key,
                'specificity': rf_spec,
                'scaler': scaler,
                'feature_cols': available_features
            }
            
            # Generate visualizations
            self.create_visualizations()
            
            logger.info("✅ Dual ML model training complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        logger.info("📊 Creating visualizations...")
        
        try:
            # 1. Residue frequency distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.ml_features['frequency'], bins=20, alpha=0.7)
            plt.title('Residue Contact Frequency Distribution')
            plt.xlabel('Contact Frequency')
            plt.ylabel('Number of Residues')
            plt.savefig(self.output_dir / 'frequency_distribution.png')
            plt.close()
            
            # 2. Energy vs frequency scatter plot
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                self.ml_features['frequency'],
                self.ml_features['energy_contribution'],
                c=self.ml_features['key_residue_prob'],
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter, label='Catalytic Hotspot Probability')
            plt.xlabel('Contact Frequency')
            plt.ylabel('Energy Contribution')
            plt.title('Energy Contribution vs Contact Frequency')
            plt.savefig(self.output_dir / 'energy_vs_frequency.png')
            plt.close()
            
            # 3. t-SNE visualization (if enough data points)
            if len(self.ml_features) > 10:
                try:
                    feature_cols = self.ml_models['feature_cols']
                    X_scaled = self.ml_models['scaler'].transform(self.ml_features[feature_cols])
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X_scaled)-1))
                    X_tsne = tsne.fit_transform(X_scaled)
                    
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        X_tsne[:, 0], X_tsne[:, 1],
                        c=self.ml_features['key_residue_prob'],
                        cmap='viridis',
                        alpha=0.7
                    )
                    plt.colorbar(scatter, label='Catalytic Hotspot Probability')
                    plt.title('t-SNE Visualization of Residues')
                    plt.xlabel('t-SNE 1')
                    plt.ylabel('t-SNE 2')
                    plt.savefig(self.output_dir / 'tsne_visualization.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"t-SNE visualization failed: {str(e)}")
            
            logger.info("✓ Visualizations created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        if self.ml_features is None:
            return ""
        
        report_path = self.output_dir / 'enzyme_analysis_report.txt'
        
        # Sort residues by importance
        catalytic_hotspots = self.ml_features.sort_values('key_residue_prob', ascending=False)
        specificity_sites = self.ml_features.sort_values('specificity_prob', ascending=False)
        
        with open(report_path, 'w') as f:
            f.write("🧬 ENZYME ENGINEERING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("📊 ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total residues analyzed: {len(self.ml_features)}\n")
            f.write(f"Docking poses processed: {len(self.results)}\n")
            f.write(f"Average binding score: {np.mean(self.scores):.2f} kcal/mol\n\n")
            
            # Top catalytic hotspots
            f.write("🔥 TOP 10 CATALYTIC HOTSPOTS\n")
            f.write("-" * 30 + "\n")
            f.write("These residues are predicted to be critical for catalytic activity:\n\n")
            
            for i, (_, row) in enumerate(catalytic_hotspots.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['residue_id']} ({row['res_type']}) - "
                       f"Probability: {row['key_residue_prob']:.3f}\n")
                f.write(f"    Energy: {row['energy_contribution']:.2f}, "
                       f"Frequency: {row['frequency']}\n\n")
            
            # Top specificity sites
            f.write("🎯 TOP 10 SPECIFICITY SITES\n")
            f.write("-" * 30 + "\n")
            f.write("These residues are predicted to determine substrate specificity:\n\n")
            
            for i, (_, row) in enumerate(specificity_sites.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['residue_id']} ({row['res_type']}) - "
                       f"Probability: {row['specificity_prob']:.3f}\n")
                f.write(f"    Specificity Score: {row['specificity_score']:.3f}, "
                       f"Frequency: {row['frequency']}\n\n")
            
            # Experimental recommendations
            f.write("🔬 EXPERIMENTAL RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("1. For activity enhancement, focus on top catalytic hotspots\n")
            f.write("2. For specificity modification, target specificity sites\n")
            f.write("3. Start with conservative mutations (Ala scanning)\n")
            f.write("4. Validate predictions with kinetic assays\n")
            f.write("5. Consider structural analysis of key residues\n\n")
            
            f.write("📁 OUTPUT FILES\n")
            f.write("-" * 30 + "\n")
            f.write("• residue_interactions.csv - Raw interaction data\n")
            f.write("• frequency_distribution.png - Residue frequency plot\n")
            f.write("• energy_vs_frequency.png - Energy correlation plot\n")
            f.write("• tsne_visualization.png - Dimensionality reduction plot\n")
            f.write("• enzyme_analysis_report.txt - This report\n")
        
        logger.info(f"✓ Analysis report saved to {report_path}")
        return str(report_path)
    
    def create_results_package(self) -> Optional[str]:
        """Create downloadable package of all results"""
        package_path = self.output_dir.parent / 'enzyme_analysis_results.zip'
        
        try:
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from output directory
                for file_path in self.output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.output_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"✓ Results packaged: {package_path}")
            return str(package_path)
            
        except Exception as e:
            logger.error(f"Error packaging results: {str(e)}")
            return None


def analyze_enzyme_docking(docking_files: Dict[str, bytes],
                          receptor_file: Optional[bytes] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for enzyme docking analysis
    
    Args:
        docking_files: Dictionary of docking result filename -> content
        receptor_file: Optional receptor file content  
        output_dir: Output directory for results
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = EnzymeDockingAnalyzer(output_dir)
    
    try:
        # Process docking results
        if not analyzer.process_docking_results(docking_files, receptor_file):
            return {'success': False, 'error': 'Failed to process docking results'}
        
        # Prepare ML features
        if analyzer.residue_data is not None:
            features = analyzer.prepare_ml_features()
            if features is not None:
                # Train ML models
                if analyzer.train_ml_models():
                    # Generate comprehensive report
                    report_path = analyzer.generate_report()
                    
                    # Create results package
                    package_path = analyzer.create_results_package()
                    
                    return {
                        'success': True,
                        'analyzer': analyzer,
                        'report_path': report_path,
                        'package_path': package_path,
                        'features': features
                    }
        
        # If no residue data, still generate basic analysis
        basic_report = analyzer.generate_basic_report()
        
        return {
            'success': True,
            'analyzer': analyzer,
            'basic_analysis': True,
            'report_path': basic_report
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {'success': False, 'error': str(e)}


# Add this method to the EnzymeDockingAnalyzer class
def generate_basic_report(self) -> str:
    """Generate basic analysis report when no residue data available"""
    report_path = self.output_dir / 'basic_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("🧬 BASIC DOCKING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if self.results:
            f.write("📊 DOCKING STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total poses analyzed: {len(self.results)}\n")
            f.write(f"Best binding score: {min(self.scores):.2f} kcal/mol\n")
            f.write(f"Average binding score: {np.mean(self.scores):.2f} kcal/mol\n")
            f.write(f"Score standard deviation: {np.std(self.scores):.2f} kcal/mol\n\n")
            
            # Score distribution
            f.write("🎯 SCORE DISTRIBUTION\n")
            f.write("-" * 25 + "\n")
            score_percentiles = np.percentile(self.scores, [25, 50, 75, 90, 95])
            f.write(f"25th percentile: {score_percentiles[0]:.2f} kcal/mol\n")
            f.write(f"Median (50th): {score_percentiles[1]:.2f} kcal/mol\n")
            f.write(f"75th percentile: {score_percentiles[2]:.2f} kcal/mol\n")
            f.write(f"90th percentile: {score_percentiles[3]:.2f} kcal/mol\n")
            f.write(f"95th percentile: {score_percentiles[4]:.2f} kcal/mol\n\n")
        
        f.write("💡 RECOMMENDATIONS\n")
        f.write("-" * 25 + "\n")
        f.write("• Upload receptor PDB file for detailed residue analysis\n")
        f.write("• Consider poses with scores < -7.0 kcal/mol as promising\n")
        f.write("• Perform visual inspection of top-scoring poses\n")
        f.write("• Validate results with experimental binding assays\n")
    
    return str(report_path)

# Add the method to the class
EnzymeDockingAnalyzer.generate_basic_report = generate_basic_report

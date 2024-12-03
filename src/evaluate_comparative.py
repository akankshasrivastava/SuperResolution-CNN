import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
import json
import seaborn as sns
from datetime import datetime

from models.model1 import SuperResolutionModel
from models.model2 import SimpleSuperResolutionModel
from data.dataset import SuperResolutionDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.results_dir = Path(config['logging']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this evaluation
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.results_dir / f"evaluation_{self.timestamp}"
        self.eval_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "model1": {"psnr": [], "ssim": [], "mse": [], "mae": []},
            "model2": {"psnr": [], "ssim": [], "mse": [], "mae": []}
        }

    def load_model(self, model_type):
        """Load model and its best checkpoint"""
        if model_type == "model1":
            model = SuperResolutionModel()
        else:
            model = SimpleSuperResolutionModel()
            
        checkpoint_path = Path(self.config['logging']['checkpoint_dir']) / model_type / 'best_model.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"Successfully loaded {model_type} model")
        except Exception as e:
            print(f"Error loading {model_type} model: {str(e)}")
            raise
        
        return model

    def tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().detach().numpy()

    def calculate_metrics(self, hr_img, sr_img):
        """Calculate all metrics between high-res and super-res images"""
        # Ensure both inputs are numpy arrays in range [0, 1]
        hr_np = np.clip(hr_img, 0, 1)
        sr_np = np.clip(sr_img, 0, 1)
        
        # Calculate PSNR
        try:
            psnr_value = psnr(hr_np, sr_np, data_range=1.0)
        except Exception as e:
            print(f"Error calculating PSNR: {str(e)}")
            psnr_value = 0.0
        
        # Calculate SSIM
        try:
            ssim_value = ssim(hr_np, sr_np, channel_axis=2, data_range=1.0)
        except Exception as e:
            print(f"Error calculating SSIM: {str(e)}")
            ssim_value = 0.0
        
        # Calculate MSE
        mse_value = np.mean((hr_np - sr_np) ** 2)
        
        # Calculate MAE
        mae_value = np.mean(np.abs(hr_np - sr_np))
        
        return {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "mse": mse_value,
            "mae": mae_value
        }

    def evaluate_model(self, model, test_loader, model_type):
        """Evaluate a single model on test dataset"""
        print(f"\nEvaluating {model_type}...")
        
        with torch.no_grad():
            for idx, (lr_img, hr_img) in enumerate(tqdm(test_loader)):
                try:
                    # Generate super-resolution image
                    lr_img = lr_img.to(self.device)
                    sr_img = model(lr_img)
                    
                    # Convert to numpy arrays
                    hr_np = hr_img.squeeze(0).permute(1, 2, 0).numpy()
                    sr_np = sr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    lr_np = lr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(hr_np, sr_np)
                    
                    # Store metrics
                    for metric_name, value in metrics.items():
                        self.metrics[model_type][metric_name].append(value)
                    
                    # Save comparison images
                    if idx < 5:
                        self.save_comparison(hr_np, sr_np, lr_np, idx, model_type, metrics)
                    
                    # Print metrics for current image
                    print(f"\nImage {idx + 1} Metrics:")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name.upper()}: {value:.4f}")
                
                except Exception as e:
                    print(f"Error processing image {idx}: {str(e)}")
                    continue

    def save_comparison(self, hr_img, sr_img, lr_img, idx, model_type, metrics):
        """Save comparison visualization"""
        try:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(hr_img)
            plt.title('High Resolution')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(lr_img)
            plt.title('Low Resolution')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(sr_img)
            plt.title(f'Super Resolution\nPSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.4f}')
            plt.axis('off')
            
            save_path = self.eval_dir / f'{model_type}_example_{idx}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved comparison image to {save_path}")
        
        except Exception as e:
            print(f"Error saving comparison image: {str(e)}")

    def plot_metrics_comparison(self):
        """Create comparative visualizations of metrics"""
        try:
            metrics_to_plot = {
                "PSNR (dB)": "psnr",
                "SSIM": "ssim",
                "MSE": "mse",
                "MAE": "mae"
            }
            
            plt.style.use('seaborn')
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
            
            for (title, metric), ax in zip(metrics_to_plot.items(), axes.flat):
                data = {
                    'Model 1': self.metrics['model1'][metric],
                    'Model 2': self.metrics['model2'][metric]
                }
                
                # Create box plot
                sns.boxplot(data=list(data.values()), ax=ax)
                ax.set_xticklabels(list(data.keys()))
                ax.set_title(title)
                
                # Add mean values
                for i, values in enumerate(data.values()):
                    mean_val = np.mean(values)
                    ax.text(i, ax.get_ylim()[0], f'μ={mean_val:.4f}', 
                           horizontalalignment='center', verticalalignment='top')
            
            plt.tight_layout()
            save_path = self.eval_dir / 'metrics_comparison.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved metrics comparison plot to {save_path}")
        
        except Exception as e:
            print(f"Error creating metrics comparison plot: {str(e)}")

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            "timestamp": self.timestamp,
            "device": str(self.device),
            "results": {}
        }
        
        for model_type in ["model1", "model2"]:
            model_metrics = {}
            for metric in ["psnr", "ssim", "mse", "mae"]:
                values = self.metrics[model_type][metric]
                if values:  # Only calculate if we have values
                    model_metrics[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }
            report["results"][model_type] = model_metrics
        
        # Save report
        try:
            report_path = self.eval_dir / 'evaluation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Saved evaluation report to {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        for model_type in ["model1", "model2"]:
            print(f"\n{model_type.upper()}:")
            for metric, stats in report["results"][model_type].items():
                print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

def main():
    try:
        # Load configuration
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create test dataset
        test_dataset = SuperResolutionDataset(
            hr_dir=Path(config['data']['processed_data_path']) / 'test' / 'hr',
            lr_dir=Path(config['data']['processed_data_path']) / 'test' / 'lr'
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Found {len(test_dataset)} test images")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Evaluate both models
        for model_type in ["model1", "model2"]:
            model = evaluator.load_model(model_type)
            evaluator.evaluate_model(model, test_loader, model_type)
        
        # Generate comparative visualizations
        evaluator.plot_metrics_comparison()
        
        # Generate and save report
        evaluator.generate_report()
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

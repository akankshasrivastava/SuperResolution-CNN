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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model

    def calculate_metrics(self, hr_img, sr_img):
        """Calculate all metrics between high-res and super-res images"""
        # Ensure images are numpy arrays and in correct range [0, 1]
        hr_img = np.clip(hr_img, 0, 1)
        sr_img = np.clip(sr_img, 0, 1)
        
        # Calculate PSNR
        psnr_value = psnr(hr_img, sr_img, data_range=1.0)
        
        # Calculate SSIM
        ssim_value = ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)
        
        # Calculate MSE
        mse_value = np.mean((hr_img - sr_img) ** 2)
        
        # Calculate MAE
        mae_value = np.mean(np.abs(hr_img - sr_img))
        
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
                # Generate super-resolution image
                lr_img = lr_img.to(self.device)
                sr_img = model(lr_img)
                
                # Convert tensors to numpy arrays
                hr_np = hr_img.squeeze(0).permute(1, 2, 0).numpy()
                sr_np = sr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                lr_np = lr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Calculate metrics
                metrics = self.calculate_metrics(hr_np, sr_np)
                
                # Store metrics
                for metric_name, value in metrics.items():
                    self.metrics[model_type][metric_name].append(value)
                
                # Save example images (first 5)
                if idx < 5:
                    self.save_comparison(hr_np, sr_np, lr_np, idx, model_type, metrics)
                    
                print(f"\nImage {idx + 1} Metrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name.upper()}: {value:.4f}")

    def save_comparison(self, hr_img, sr_img, lr_img, idx, model_type, metrics):
        """Save comparison visualization"""
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
        
        plt.savefig(self.eval_dir / f'{model_type}_example_{idx}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_metrics_comparison(self):
        """Create comparative visualizations of metrics"""
        metrics_to_plot = {
            "PSNR (dB)": "psnr",
            "SSIM": "ssim",
            "MSE": "mse",
            "MAE": "mae"
        }
        
        # Set style
        plt.style.use('seaborn')
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        
        for (title, metric), ax in zip(metrics_to_plot.items(), axes.flat):
            data = {
                'Model 1': self.metrics['model1'][metric],
                'Model 2': self.metrics['model2'][metric]
            }
            
            # Create box plot instead of violin plot for better visualization
            sns.boxplot(data=list(data.values()), ax=ax)
            ax.set_xticklabels(list(data.keys()))
            ax.set_title(title)
            
            # Add mean values as text
            for i, values in enumerate(data.values()):
                mean_val = np.mean(values)
                ax.text(i, ax.get_ylim()[0], f'μ={mean_val:.4f}', 
                       horizontalalignment='center', verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'metrics_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

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
                model_metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            report["results"][model_type] = model_metrics
        
        # Save report
        with open(self.eval_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        for model_type in ["model1", "model2"]:
            print(f"\n{model_type.upper()}:")
            for metric, stats in report["results"][model_type].items():
                print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

def main():
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

if __name__ == "__main__":
    main()

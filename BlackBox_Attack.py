import os
import shlex
import json
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import random
from tqdm import tqdm
from phishintention import PhishIntentionWrapper
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ImageQualityMetrics:
    @staticmethod
    def calculate_psnr(original, modified):
        """Calculate Peak Signal-to-Noise Ratio between two images"""
        return psnr(original, modified)
    
    @staticmethod
    def calculate_ssim(original, modified):
        """Calculate Structural Similarity Index between two images"""
        return ssim(original, modified, channel_axis=2, data_range=modified.max() - modified.min())
    
    @staticmethod
    def calculate_l2_distance(original, modified):
        """Calculate L2 (Euclidean) distance between images"""
        return np.sqrt(np.mean((original - modified) ** 2))

class AttackVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_data = {
            'jpeg': {'psnr': [], 'ssim': [], 'l2': []},
            'uap': {'psnr': [], 'ssim': [], 'l2': []},
            'spatial': {'psnr': [], 'ssim': [], 'l2': []},
            'color': {'psnr': [], 'ssim': [], 'l2': []}
        }
        self.success_data = {attack: {'success': 0, 'total': 0} for attack in self.metrics_data.keys()}
        
    def update_metrics(self, attack_type, original_img, modified_img):
        """Update metrics for a given attack type"""
        metrics = {
            'psnr': ImageQualityMetrics.calculate_psnr(original_img, modified_img),
            'ssim': ImageQualityMetrics.calculate_ssim(original_img, modified_img),
            'l2': ImageQualityMetrics.calculate_l2_distance(original_img, modified_img)
        }
        
        for metric_name, value in metrics.items():
            self.metrics_data[attack_type][metric_name].append(value)
    
    def update_success_rate(self, attack_type, is_successful):
        """Update success rate for a given attack type"""
        self.success_data[attack_type]['total'] += 1
        if is_successful:
            self.success_data[attack_type]['success'] += 1
    
    def plot_metrics_distribution(self):
        """Plot distribution of quality metrics for each attack type"""
        metrics = ['psnr', 'ssim', 'l2']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            data = []
            labels = []
            for attack in self.metrics_data.keys():
                data.extend(self.metrics_data[attack][metric])
                labels.extend([attack.upper()] * len(self.metrics_data[attack][metric]))
            
            sns.boxplot(x=labels, y=data, ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()} Distribution')
            axes[idx].set_xlabel('Attack Type')
            axes[idx].set_ylabel(metric.upper())
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_distribution.png'))
        plt.close()
    
    def plot_success_rates(self):
        """Plot success rates for each attack type"""
        attacks = list(self.success_data.keys())
        success_rates = [
            (self.success_data[attack]['success'] / self.success_data[attack]['total']) * 100 
            if self.success_data[attack]['total'] > 0 else 0 
            for attack in attacks
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([attack.upper() for attack in attacks], success_rates)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.title('Attack Success Rates')
        plt.xlabel('Attack Type')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        
        plt.savefig(os.path.join(self.output_dir, 'success_rates.png'))
        plt.close()
    
    def plot_metrics_radar(self):
        """Create a radar plot comparing average metrics across attack types"""
        metrics = ['psnr', 'ssim', 'l2']
        attacks = list(self.metrics_data.keys())
        
        # Calculate averages
        avg_metrics = {attack: [] for attack in attacks}
        for attack in attacks:
            for metric in metrics:
                values = self.metrics_data[attack][metric]
                avg_metrics[attack].append(np.mean(values) if values else 0)
        
        # Normalize metrics to 0-1 scale for radar plot
        normalized_metrics = {attack: [] for attack in attacks}
        for metric_idx in range(len(metrics)):
            values = [avg_metrics[attack][metric_idx] for attack in attacks]
            min_val = min(values)
            max_val = max(values)
            for attack in attacks:
                if max_val - min_val != 0:
                    normalized_val = (avg_metrics[attack][metric_idx] - min_val) / (max_val - min_val)
                else:
                    normalized_val = 0
                normalized_metrics[attack].append(normalized_val)
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for attack in attacks:
            values = normalized_metrics[attack]
            values = np.concatenate((values, [values[0]]))  # complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=attack.upper())
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Normalized Metrics Comparison')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.savefig(os.path.join(self.output_dir, 'metrics_radar.png'))
        plt.close()

class BlackBoxAttacks:
    def __init__(self, input_dir, output_dir, num_variations=10, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.num_variations = num_variations
        self.visualizer = AttackVisualizer(output_dir)

    def get_image_transforms(self, image_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform

    def setup_output_directories(self, site_folder):
        image_dir = os.path.join(self.output_dir, site_folder)
        os.makedirs(image_dir, exist_ok=True)
        
        attack_dirs = {}
        for attack in ['jpeg', 'uap', 'spatial', 'color']:
            attack_dir = os.path.join(image_dir, attack)
            os.makedirs(attack_dir, exist_ok=True)
            attack_dirs[attack] = attack_dir
            
        return attack_dirs

    def jpeg_compression_attack(self, image, variation_id):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        quality = random.randint(5, 20)
        
        temp_path = f"temp_{variation_id}.jpg"
        image.save(temp_path, "JPEG", quality=quality)
        compressed = Image.open(temp_path)
        compressed = compressed.resize((image.width, image.height), Image.Resampling.LANCZOS)
        os.remove(temp_path)
        
        return np.array(compressed), {"quality": quality}

    def generate_universal_perturbation(self, image_size, epsilon=None):
        if epsilon is None:
            epsilon = random.uniform(0, 0.5)
        
        perturbation = torch.zeros((3, image_size[1], image_size[0])).to(self.device)
        perturbation = torch.rand_like(perturbation) * 2 * epsilon - epsilon
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        
        return perturbation, {"epsilon": epsilon}

    def apply_uap(self, image, image_tensor, variation_id):
        image_size = (image.width, image.height)
        perturbation, params = self.generate_universal_perturbation(image_size)
        
        image_tensor = image_tensor.to(self.device)
        perturbed = image_tensor + perturbation
        perturbed = torch.clamp(perturbed, 0, 1)
        
        perturbed_image = transforms.ToPILImage()(perturbed.cpu())
        perturbed_image = perturbed_image.resize(image_size, Image.Resampling.LANCZOS)
        
        return np.array(perturbed_image), params

    def spatial_transformation_attack(self, image, variation_id):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_size = (image.width, image.height)
        transform = self.get_image_transforms(image_size)
            
        max_shift = min(image_size) // 8
        max_rotation = random.uniform(10, 20)
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        angle = random.uniform(-max_rotation, max_rotation)
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        theta = torch.tensor([
            [1, 0, shift_x / (image_size[0] / 2)],
            [0, 1, shift_y / (image_size[1] / 2)]
        ], dtype=torch.float).unsqueeze(0).to(self.device)
        
        grid = F.affine_grid(theta, image_tensor.size(), align_corners=True)
        transformed = F.grid_sample(image_tensor, grid, align_corners=True)
        
        transformed_image = transforms.ToPILImage()(transformed.squeeze(0).cpu())
        transformed_image = transformed_image.resize(image_size, Image.Resampling.LANCZOS)
        
        params = {
            "shift_x": shift_x,
            "shift_y": shift_y,
            "angle": angle
        }
        return np.array(transformed_image), params

    def color_manipulation_attack(self, image, variation_id):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image_size = (image.width, image.height)
        transform = self.get_image_transforms(image_size)
            
        intensity = random.uniform(0.05, 0.15)
        hue = random.uniform(-intensity, intensity)
        saturation = random.uniform(1-intensity, 1+intensity)
        brightness = random.uniform(1-intensity, 1+intensity)
        contrast = random.uniform(1-intensity, 1+intensity)
        
        image_tensor = transform(image).to(self.device)
        
        image_tensor = transforms.functional.adjust_hue(image_tensor, hue)
        image_tensor = transforms.functional.adjust_saturation(image_tensor, saturation)
        image_tensor = transforms.functional.adjust_brightness(image_tensor, brightness)
        image_tensor = transforms.functional.adjust_contrast(image_tensor, contrast)
        
        manipulated_image = transforms.ToPILImage()(image_tensor.cpu())
        manipulated_image = manipulated_image.resize(image_size, Image.Resampling.LANCZOS)
        
        params = {
            "hue": hue,
            "saturation": saturation,
            "brightness": brightness,
            "contrast": contrast
        }
        return np.array(manipulated_image), params

    def save_attack_results(self, image, params, output_dir, attack_type, variation_id):
        image_path = os.path.join(output_dir, f'{attack_type}_variation_{variation_id}.png')
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(image_path)
        else:
            image.save(image_path)
        
        params_path = os.path.join(output_dir, f'{attack_type}_variation_{variation_id}_params.txt')
        with open(params_path, 'w') as f:
            f.write(f"Original image size: {image.shape if isinstance(image, np.ndarray) else (image.width, image.height)}\n")
            for key, value in params.items():
                f.write(f'{key}: {value}\n')

    def process_dataset(self, num_samples=1000):
        site_folders = [f for f in os.listdir(self.input_dir) 
                       if os.path.isdir(os.path.join(self.input_dir, f))]
        
        valid_folders = []
        for folder in site_folders:
            if os.path.exists(os.path.join(self.input_dir, folder, 'shot.png')):
                valid_folders.append(folder)
        
        total_samples = len(valid_folders)
        if total_samples < num_samples:
            print(f"Warning: Only {total_samples} valid samples available, using all of them.")
            selected_folders = valid_folders
        else:
            selected_folders = random.sample(valid_folders, num_samples)
            print(f"Randomly selected {num_samples} samples from {total_samples} total samples.")
        
        sample_info_path = os.path.join(self.output_dir, 'sample_selection_info.txt')
        os.makedirs(self.output_dir, exist_ok=True)
        
        site_pbar = tqdm(selected_folders, desc="Processing sites", position=0)
        
        # Dictionary to store timing information
        site_timings = {}
        
        for site_folder in site_pbar:
            site_start_time = time.time()
            site_pbar.set_description(f"Processing site: {site_folder}")
            
            img_path = os.path.join(self.input_dir, site_folder, 'shot.png')
            
            try:
                original_image = Image.open(img_path).convert('RGB')
                original_np = np.array(original_image)
                
                attack_dirs = self.setup_output_directories(site_folder)
                image_size = (original_image.width, original_image.height)
                transform = self.get_image_transforms(image_size)
                image_tensor = transform(original_image)
                
                variation_pbar = tqdm(range(self.num_variations), 
                                    desc="Generating variations",
                                    position=1,
                                    leave=False)
                
                for i in variation_pbar:
                    # Record start time for each attack type
                    attack_times = {}
                    
                    # JPEG Attack
                    start_time = time.time()
                    jpeg_attacked, jpeg_params = self.jpeg_compression_attack(original_image, i)
                    self.visualizer.update_metrics('jpeg', original_np, jpeg_attacked)
                    self.save_attack_results(jpeg_attacked, jpeg_params, 
                                          attack_dirs['jpeg'], 'jpeg', i)
                    attack_times['jpeg'] = time.time() - start_time
                    
                    # UAP Attack
                    start_time = time.time()
                    uap_attacked, uap_params = self.apply_uap(original_image, image_tensor, i)
                    self.visualizer.update_metrics('uap', original_np, uap_attacked)
                    self.save_attack_results(uap_attacked, uap_params, 
                                          attack_dirs['uap'], 'uap', i)
                    attack_times['uap'] = time.time() - start_time
                    
                    # Spatial Attack
                    start_time = time.time()
                    spatial_attacked, spatial_params = self.spatial_transformation_attack(original_image, i)
                    self.visualizer.update_metrics('spatial', original_np, spatial_attacked)
                    self.save_attack_results(spatial_attacked, spatial_params, 
                                          attack_dirs['spatial'], 'spatial', i)
                    attack_times['spatial'] = time.time() - start_time
                    
                    # Color Attack
                    start_time = time.time()
                    color_attacked, color_params = self.color_manipulation_attack(original_image, i)
                    self.visualizer.update_metrics('color', original_np, color_attacked)
                    self.save_attack_results(color_attacked, color_params, 
                                          attack_dirs['color'], 'color', i)
                    attack_times['color'] = time.time() - start_time
                    
                variation_pbar.close()
                
                # Record total site processing time
                site_time = time.time() - site_start_time
                site_timings[site_folder] = {
                    'total_time': site_time,
                    'attack_times': attack_times
                }
                
                print(f"\nSite {site_folder} completed in {site_time:.2f} seconds")
                print("Attack type timings:")
                for attack_type, attack_time in attack_times.items():
                    print(f"- {attack_type.upper()}: {attack_time:.2f} seconds")
                
            except Exception as e:
                print(f"\nError processing {site_folder}: {str(e)}")
                continue
        
        site_pbar.close()
        
        # Print timing summary
        print("\nTiming Summary for All Sites:")
        print("="*50)
        total_time = sum(timing['total_time'] for timing in site_timings.values())
        avg_time = total_time / len(site_timings) if site_timings else 0
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per site: {avg_time:.2f} seconds")
        print("\nAverage time per attack type:")
        
        attack_types = ['jpeg', 'uap', 'spatial', 'color']
        for attack_type in attack_types:
            avg_attack_time = sum(
                timing['attack_times'].get(attack_type, 0) 
                for timing in site_timings.values()
            ) / len(site_timings) if site_timings else 0
            print(f"- {attack_type.upper()}: {avg_attack_time:.2f} seconds")
        
        # Save timing information
        timing_file = os.path.join(self.output_dir, 'processing_times.json')
        with open(timing_file, 'w') as f:
            json.dump({
                'site_timings': site_timings,
                'total_time': total_time,
                'average_time_per_site': avg_time
            }, f, indent=2)
        
        # Generate visualization plots
        print("\nGenerating statistical visualizations...")
        self.visualizer.plot_metrics_distribution()
        self.visualizer.plot_metrics_radar()
        print("Statistical visualizations have been saved to the output directory.")

def save_attack_summary_txt(output_dir, attack_summary, timing_summary, total_testing_time, total_attacks, successful_attacks, attack_type_stats):
    """
    Save attack summary to a text file.
    """
    summary_file = os.path.join(output_dir, 'Blackbox_attack_report.txt')
    
    with open(summary_file, 'w') as f:
        # Write header
        f.write("="*50 + "\n")
        f.write("ATTACK RESULTS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Overall Statistics
        f.write("Overall Statistics:\n")
        f.write(f"- Total testing time: {total_testing_time:.2f} seconds ({total_testing_time/60:.2f} minutes)\n")
        f.write(f"- Total attacks performed: {total_attacks}\n")
        f.write(f"- Successful attacks: {successful_attacks}\n")
        if total_attacks > 0:
            f.write(f"- Overall success rate: {successful_attacks / total_attacks:.2%}\n")
        f.write("\n")
        
        # Per-Site Timing Summary
        f.write("Per-Site Timing Summary:\n")
        for site, timing in timing_summary.items():
            f.write(f"\nSite: {site}\n")
            f.write(f"- Total time: {timing['total']:.2f} seconds\n")
            f.write("- Attack timings:\n")
            for attack, attack_timing in timing["attacks"].items():
                f.write(f"  * {attack.upper()}: {attack_timing['total']:.2f} seconds\n")
        f.write("\n")
        
        # Per-Attack Statistics
        f.write("Per-Attack Statistics:\n")
        for attack in attack_type_stats:
            stats = attack_type_stats[attack]
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"\n{attack.upper()}:\n")
            f.write(f"- Total attempts: {stats['total']}\n")
            f.write(f"- Successful: {stats['success']}\n")
            f.write(f"- Success rate: {success_rate:.2%}\n")
            
            # Calculate average timing for this attack type
            attack_times = [site_timing["attacks"].get(attack, {}).get("total", 0) 
                           for site_timing in timing_summary.values()]
            avg_time = sum(attack_times) / len(attack_times) if attack_times else 0
            f.write(f"- Average time per site: {avg_time:.2f} seconds\n")
        f.write("\n")
        
        # Per-Site Attack Success Summary
        f.write("Per-Site Attack Success Summary:\n")
        for site, summary in attack_summary.items():
            f.write(f"\nSite: {site}\n")
            f.write(f"Original Prediction: {summary['original_pred']}\n")
            f.write(f"URL: {summary['url']}\n")
            if summary['attacks']:
                f.write("Successful Attacks:\n")
                for attack_type, variations in summary['attacks'].items():
                    f.write(f"- {attack_type.upper()}: {len(variations)} successful variations\n")
            else:
                f.write("No successful attacks\n")
        
    print(f"\nDetailed attack summary saved to {summary_file}")
    return summary_file

def test_blackbox_attacks(attack_dir, original_dir, phishintention, output_file):
    """
    Test black box attacks and generate comprehensive statistics and visualizations.
    """
    # Initialize counters and data structures
    total_attacks = 0
    successful_attacks = 0
    attack_summary = {}
    timing_summary = {}
    attack_type_stats = {
        "jpeg": {"total": 0, "success": 0},
        "uap": {"total": 0, "success": 0},
        "spatial": {"total": 0, "success": 0},
        "color": {"total": 0, "success": 0}
    }
    visualizer = AttackVisualizer(attack_dir)
    
    # Start overall testing timer
    testing_start_time = time.time()
    
    # Get only directories from attack_dir
    sites = [f for f in os.listdir(attack_dir) 
             if os.path.isdir(os.path.join(attack_dir, f))]
    
    # Main processing loop
    for site_folder in tqdm(sites, desc="Testing attacks", position=0):
        # Start site timer
        site_start_time = time.time()
        timing_summary[site_folder] = {"total": 0, "attacks": {}}
        
        original_screenshot = os.path.join(original_dir, site_folder, "shot.png")
        original_info = os.path.join(original_dir, site_folder, "info.txt")
        
        # Skip if original screenshot doesn't exist
        if not os.path.exists(original_screenshot):
            print(f"Skipping {site_folder} - original screenshot not found")
            continue

        # Get URL from info file or construct from folder name
        if not os.path.exists(original_info):
            url = "https://" + site_folder
        else:
            with open(original_info) as f:
                url = f.read().strip()

        # Get original prediction
        try:
            original_pred_start = time.time()
            original_pred, *_ = phishintention.test_orig_phishintention(url, original_screenshot)
            original_pred_time = time.time() - original_pred_start
            timing_summary[site_folder]["original_prediction"] = original_pred_time
            
        except Exception as e:
            print(f"Error getting original prediction for {site_folder}: {str(e)}")
            continue

        # Initialize site summary
        site_summary = {
            "original_pred": original_pred,
            "url": url,
            "attacks": {}
        }

        # Process each attack type
        for attack in ["jpeg", "uap", "spatial", "color"]:
            # Start attack timer
            attack_start_time = time.time()
            timing_summary[site_folder]["attacks"][attack] = {"total": 0, "variations": {}}
            
            attack_dir_path = os.path.join(attack_dir, site_folder, attack)
            attack_tqdm = tqdm(desc=f"Testing {attack.upper()}", position=1, leave=False)
            
            # Skip if attack directory doesn't exist or is not a directory
            if not (os.path.exists(attack_dir_path) and os.path.isdir(attack_dir_path)):
                print(f"Skipping {attack_dir_path} - not a valid directory")
                continue
            
            # Get variations for this attack
            variations = [v for v in os.listdir(attack_dir_path) 
                        if v.endswith(".png") and v != "original.png"]
            
            # Process each variation
            for variation in variations:
                # Start variation timer
                variation_start_time = time.time()
                
                variation_path = os.path.join(attack_dir_path, variation)
                variation_info = os.path.join(attack_dir_path, "info.txt")
                
                # Get URL for variation
                if not os.path.exists(variation_info):
                    test_url = url
                else:
                    with open(variation_info) as f:
                        test_url = f.read().strip()
                
                try:
                    # Test attack
                    attacked_pred, *_ = phishintention.test_orig_phishintention(test_url, variation_path)
                    
                    # Update statistics
                    total_attacks += 1
                    attack_type_stats[attack]["total"] += 1
                    is_successful = attacked_pred != original_pred
                    
                    if is_successful:
                        successful_attacks += 1
                        attack_type_stats[attack]["success"] += 1
                        site_summary["attacks"].setdefault(attack, {}).update({
                            variation: {
                                "prediction": attacked_pred,
                                "original_prediction": original_pred
                            }
                        })
                    
                    # Update visualizer
                    visualizer.update_success_rate(attack, is_successful)
                    
                    # Record variation timing
                    variation_time = time.time() - variation_start_time
                    timing_summary[site_folder]["attacks"][attack]["variations"][variation] = variation_time
                    timing_summary[site_folder]["attacks"][attack]["total"] += variation_time
                    
                except Exception as e:
                    print(f"Error processing {variation_path}: {str(e)}")
                    continue
                
                attack_tqdm.update(1)
            
            # Record attack timing
            attack_time = time.time() - attack_start_time
            timing_summary[site_folder]["attacks"][attack]["total"] = attack_time
            print(f"\n{site_folder} - {attack.upper()} attack completed in {attack_time:.2f} seconds")
            
            attack_tqdm.close()
        
        # Record total site timing
        site_time = time.time() - site_start_time
        timing_summary[site_folder]["total"] = site_time
        print(f"\n{site_folder} - Total testing time: {site_time:.2f} seconds")
        
        # Add site summary to overall summary
        attack_summary[site_folder] = site_summary
    
    # Record total testing time
    total_testing_time = time.time() - testing_start_time
    
    # Generate and save visualizations
    try:
        visualizer.plot_success_rates()
        print("Success rate visualization saved successfully")
    except Exception as e:
        print(f"Error generating success rate visualization: {str(e)}")
    
    # Save detailed summary to JSON and TXT
    try:
        # Save JSON summary
        with open(output_file, "w") as f:
            json.dump({
                "attack_summary": attack_summary,
                "timing_summary": timing_summary,
                "total_testing_time": total_testing_time
            }, f, indent=2)
        print(f"Attack summary saved to {output_file}")
        
        # Save TXT summary
        txt_file = save_attack_summary_txt(
            attack_dir,
            attack_summary,
            timing_summary,
            total_testing_time,
            total_attacks,
            successful_attacks,
            attack_type_stats
        )
    except Exception as e:
        print(f"Error saving attack summary: {str(e)}")
    
    # Print comprehensive statistics
    print("\n" + "="*50)
    print("ATTACK RESULTS SUMMARY")
    print("="*50)
    print(f"\nOverall Statistics:")
    print(f"- Total testing time: {total_testing_time:.2f} seconds")
    print(f"- Total attacks performed: {total_attacks}")
    print(f"- Successful attacks: {successful_attacks}")
    if total_attacks > 0:
        print(f"- Overall success rate: {successful_attacks / total_attacks:.2%}")
    
    print("\nPer-Site Timing Summary:")
    for site, timing in timing_summary.items():
        print(f"\nSite: {site}")
        print(f"- Total time: {timing['total']:.2f} seconds")
        print("- Attack timings:")
        for attack, attack_timing in timing["attacks"].items():
            print(f"  * {attack.upper()}: {attack_timing['total']:.2f} seconds")
    
    print("\nPer-Attack Statistics:")
    for attack in attack_type_stats:
        stats = attack_type_stats[attack]
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"\n{attack.upper()}:")
        print(f"- Total attempts: {stats['total']}")
        print(f"- Successful: {stats['success']}")
        print(f"- Success rate: {success_rate:.2%}")
        
        # Calculate average timing for this attack type
        attack_times = [site_timing["attacks"].get(attack, {}).get("total", 0) 
                       for site_timing in timing_summary.values()]
        avg_time = sum(attack_times) / len(attack_times) if attack_times else 0
        print(f"- Average time per site: {avg_time:.2f} seconds")
    
    print("\nPer-Site Statistics:")
    successful_sites = sum(1 for site in attack_summary.values() if any(site["attacks"].values()))
    print(f"- Total sites tested: {len(sites)}")
    print(f"- Sites with successful attacks: {successful_sites}")
    if len(sites) > 0:
        print(f"- Site success rate: {successful_sites / len(sites):.2%}")
    
    print("\nOutput Files:")
    print(f"- Success rate plot: {os.path.join(attack_dir, 'success_rates.png')}")
    print(f"- Detailed JSON summary: {output_file}")
    """
    Test black box attacks and generate comprehensive statistics and visualizations.
    
    Args:
        attack_dir: Directory containing attack results
        original_dir: Directory containing original screenshots
        phishintention: PhishIntention model wrapper
        output_file: Path to save the JSON summary
    """
    # Initialize counters and data structures
    total_attacks = 0
    successful_attacks = 0
    attack_summary = {}
    attack_type_stats = {
        "jpeg": {"total": 0, "success": 0},
        "uap": {"total": 0, "success": 0},
        "spatial": {"total": 0, "success": 0},
        "color": {"total": 0, "success": 0}
    }
    visualizer = AttackVisualizer(attack_dir)
    
    # Get only directories from attack_dir
    sites = [f for f in os.listdir(attack_dir) 
             if os.path.isdir(os.path.join(attack_dir, f))]
    
    # Main processing loop
    for site_folder in tqdm(sites, desc="Testing attacks", position=0):
        original_screenshot = os.path.join(original_dir, site_folder, "shot.png")
        original_info = os.path.join(original_dir, site_folder, "info.txt")
        
        # Skip if original screenshot doesn't exist
        if not os.path.exists(original_screenshot):
            print(f"Skipping {site_folder} - original screenshot not found")
            continue

        # Get URL from info file or construct from folder name
        if not os.path.exists(original_info):
            url = "https://" + site_folder
        else:
            with open(original_info) as f:
                url = f.read().strip()

        # Get original prediction
        try:
            original_pred, *_ = phishintention.test_orig_phishintention(url, original_screenshot)
        except Exception as e:
            print(f"Error getting original prediction for {site_folder}: {str(e)}")
            continue

        # Initialize site summary
        site_summary = {
            "original_pred": original_pred,
            "url": url,
            "attacks": {}
        }

        # Process each attack type
        for attack in ["jpeg", "uap", "spatial", "color"]:
            attack_dir_path = os.path.join(attack_dir, site_folder, attack)
            attack_tqdm = tqdm(desc=f"Testing {attack.upper()}", position=1, leave=False)
            
            # Skip if attack directory doesn't exist or is not a directory
            if not (os.path.exists(attack_dir_path) and os.path.isdir(attack_dir_path)):
                print(f"Skipping {attack_dir_path} - not a valid directory")
                continue
            
            # Get variations for this attack
            variations = [v for v in os.listdir(attack_dir_path) 
                        if v.endswith(".png") and v != "original.png"]
            
            # Process each variation
            for variation in variations:
                variation_path = os.path.join(attack_dir_path, variation)
                variation_info = os.path.join(attack_dir_path, "info.txt")
                
                # Get URL for variation
                if not os.path.exists(variation_info):
                    test_url = url
                else:
                    with open(variation_info) as f:
                        test_url = f.read().strip()
                
                try:
                    # Test attack
                    attacked_pred, *_ = phishintention.test_orig_phishintention(test_url, variation_path)
                    
                    # Update statistics
                    total_attacks += 1
                    attack_type_stats[attack]["total"] += 1
                    is_successful = attacked_pred != original_pred
                    
                    if is_successful:
                        successful_attacks += 1
                        attack_type_stats[attack]["success"] += 1
                        site_summary["attacks"].setdefault(attack, {}).update({
                            variation: {
                                "prediction": attacked_pred,
                                "original_prediction": original_pred
                            }
                        })
                    
                    # Update visualizer
                    visualizer.update_success_rate(attack, is_successful)
                    
                except Exception as e:
                    print(f"Error processing {variation_path}: {str(e)}")
                    continue
                
                attack_tqdm.update(1)
            
            attack_tqdm.close()
        
        # Add site summary to overall summary
        attack_summary[site_folder] = site_summary
    
    # Generate and save visualizations
    try:
        visualizer.plot_success_rates()
        print("Success rate visualization saved successfully")
    except Exception as e:
        print(f"Error generating success rate visualization: {str(e)}")
    
    # Save detailed summary to JSON
    try:
        with open(output_file, "w") as f:
            json.dump(attack_summary, f, indent=2)
        print(f"Attack summary saved to {output_file}")
    except Exception as e:
        print(f"Error saving attack summary: {str(e)}")
    
    # Print comprehensive statistics
    print("\n" + "="*50)
    print("ATTACK RESULTS SUMMARY")
    print("="*50)
    print(f"\nOverall Statistics:")
    print(f"- Total attacks performed: {total_attacks}")
    print(f"- Successful attacks: {successful_attacks}")
    if total_attacks > 0:
        print(f"- Overall success rate: {successful_attacks / total_attacks:.2%}")
    
    print("\nPer-Attack Statistics:")
    for attack in attack_type_stats:
        stats = attack_type_stats[attack]
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"- {attack.upper()}:")
        print(f"  * Total attempts: {stats['total']}")
        print(f"  * Successful: {stats['success']}")
        print(f"  * Success rate: {success_rate:.2%}")
    
    print("\nPer-Site Statistics:")
    successful_sites = sum(1 for site in attack_summary.values() if any(site["attacks"].values()))
    print(f"- Total sites tested: {len(sites)}")
    print(f"- Sites with successful attacks: {successful_sites}")
    if len(sites) > 0:
        print(f"- Site success rate: {successful_sites / len(sites):.2%}")
    
    print("\nVisualization files:")
    print(f"- Success rate plot: {os.path.join(attack_dir, 'success_rates.png')}")
    print(f"- Detailed JSON summary: {output_file}")

def main():
    input_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/datasets/test_sites"
    output_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/BlackBox_attack_results3"
    
    # Start total execution timer
    total_start_time = time.time()
    
    print("\nStarting attack generation and testing process...")
    print("="*50)
    
    attacker = BlackBoxAttacks(
        input_dir=input_directory,
        output_dir=output_directory,
        num_variations=5
    )
    
    # Track attack generation time
    generation_start_time = time.time()
    attacker.process_dataset(num_samples=5)
    generation_time = time.time() - generation_start_time
    
    print(f"\nAttack generation completed in {generation_time:.2f} seconds!")
    
    # Track testing time
    print("\nStarting attack testing...")
    testing_start_time = time.time()
    output_file = "attack_summary.json"
    phishintention = PhishIntentionWrapper()
    test_blackbox_attacks(output_directory, input_directory, phishintention, output_file)
    testing_time = time.time() - testing_start_time
    
    # Calculate and print total execution time
    total_execution_time = time.time() - total_start_time
    
    print("\n" + "="*50)
    print("EXECUTION TIME SUMMARY")
    print("="*50)
    print(f"Attack Generation Time: {generation_time/60:.2f} minutes")
    print(f"Attack Testing Time: {testing_time/60:.2f} minutes")
    print(f"Total Execution Time: {total_execution_time/60:.2f} minutes")
    
if __name__ == "__main__":
    main()
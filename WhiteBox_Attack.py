import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os
from tqdm import tqdm
import json
from datetime import datetime
from phishintention import PhishIntentionWrapper
from modules.logo_matching import ocr_main
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pixel_difference_heatmap(original_path, perturbed_path, output_path):
    """
    Create a heatmap showing the pixel-wise differences between original and perturbed images
    """
    # Load images
    original = np.array(Image.open(original_path))
    perturbed = np.array(Image.open(perturbed_path))
    
    # Calculate absolute difference
    diff = np.abs(original - perturbed)
    diff_magnitude = np.mean(diff, axis=2)  # Average across color channels
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(diff_magnitude, cmap='YlOrRd')
    plt.title('Pixel Differences Heatmap')
    plt.colorbar(label='Difference Magnitude')
    plt.savefig(output_path)
    plt.close()

def plot_confidence_comparison(original_conf, fgsm_conf, pgd_conf, output_path):
    """
    Create a bar plot comparing confidence scores
    """
    plt.figure(figsize=(10, 6))
    methods = ['Original', 'FGSM Attack', 'PGD Attack']
    confidences = [original_conf, fgsm_conf, pgd_conf]
    
    bars = plt.bar(methods, confidences)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.title('Confidence Score Comparison')
    plt.ylabel('Confidence Score (%)')
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()

def plot_attack_success_rates(metrics_summary, output_path):
    """
    Create a grouped bar plot showing success rates for different attack types
    """
    plt.figure(figsize=(12, 6))
    
    attack_types = ['FGSM', 'PGD']
    metrics = ['overall', 'category_change', 'confidence_impact', 'target_change']
    labels = ['Overall', 'Category Change', 'Confidence Impact', 'Target Change']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Extract success rates
    fgsm_rates = [metrics_summary['attack_success_rates']['fgsm'][m] for m in metrics]
    pgd_rates = [metrics_summary['attack_success_rates']['pgd'][m] for m in metrics]
    
    # Create grouped bars
    plt.bar(x - width/2, [rate * 100 for rate in fgsm_rates], width, label='FGSM')
    plt.bar(x + width/2, [rate * 100 for rate in pgd_rates], width, label='PGD')
    
    plt.ylabel('Success Rate (%)')
    plt.title('Attack Success Rates by Type')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(fgsm_rates):
        plt.text(i - width/2, v * 100 + 1, f'{v*100:.1f}%', ha='center')
    for i, v in enumerate(pgd_rates):
        plt.text(i + width/2, v * 100 + 1, f'{v*100:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confidence_impact_distribution(all_results, output_path):
    """
    Create violin plots showing the distribution of confidence impacts
    """
    fgsm_impacts = []
    pgd_impacts = []
    
    for site_results in all_results.values():
        if 'attacks' in site_results:
            orig_conf = site_results['original']['confidence']
            
            if 'fgsm' in site_results['attacks']:
                fgsm_conf = site_results['attacks']['fgsm'].get('confidence')
                if orig_conf is not None and fgsm_conf is not None:
                    fgsm_impacts.append(orig_conf - fgsm_conf)
            
            if 'pgd' in site_results['attacks']:
                pgd_conf = site_results['attacks']['pgd'].get('confidence')
                if orig_conf is not None and pgd_conf is not None:
                    pgd_impacts.append(orig_conf - pgd_conf)
    
    plt.figure(figsize=(10, 6))
    data = [fgsm_impacts, pgd_impacts]
    
    parts = plt.violinplot(data, showmeans=True)
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2], ['FGSM', 'PGD'])
    plt.ylabel('Confidence Score Reduction')
    plt.title('Distribution of Attack Impact on Confidence Scores')
    
    plt.savefig(output_path)
    plt.close()

def generate_attack_visualizations(site_folder, site_results, site_output_dir):
    """
    Generate all visualizations for a single site
    """
    # Original and perturbed image paths
    orig_path = os.path.join(site_output_dir, "original_vis.png")
    fgsm_path = os.path.join(site_output_dir, "fgsm_perturbed.png")
    pgd_path = os.path.join(site_output_dir, "pgd_perturbed.png")
    
    # Generate heatmaps for both attacks
    if os.path.exists(orig_path):
        if os.path.exists(fgsm_path):
            plot_pixel_difference_heatmap(
                orig_path, 
                fgsm_path, 
                os.path.join(site_output_dir, "fgsm_difference_heatmap.png")
            )
        if os.path.exists(pgd_path):
            plot_pixel_difference_heatmap(
                orig_path,
                pgd_path,
                os.path.join(site_output_dir, "pgd_difference_heatmap.png")
            )
    
    # Plot confidence comparison
    orig_conf = site_results['original'].get('confidence', 0)
    fgsm_conf = site_results['attacks'].get('fgsm', {}).get('confidence', 0)
    pgd_conf = site_results['attacks'].get('pgd', {}).get('confidence', 0)
    
    plot_confidence_comparison(
        orig_conf,
        fgsm_conf,
        pgd_conf,
        os.path.join(site_output_dir, "confidence_comparison.png")
    )

def add_visualizations_to_process_test_directory():
    """
    Add the following code at the end of process_test_directory function, 
    just before returning output_dir and results_with_metrics:
    """
    # Generate visualizations for attack success rates
    plot_attack_success_rates(
        metrics_summary,
        os.path.join(output_dir, 'attack_success_rates.png')
    )
    
    # Generate visualization for confidence impact distribution
    plot_confidence_impact_distribution(
        all_results,
        os.path.join(output_dir, 'confidence_impact_distribution.png')
    )
    
    # Generate per-site visualizations
    for site_folder, site_results in all_results.items():
        site_output_dir = os.path.join(output_dir, site_folder)
        generate_attack_visualizations(site_folder, site_results, site_output_dir)

class PhishIntentionAttack:
    def __init__(self, phishintention_wrapper, epsilon=0.1):
        self.wrapper = phishintention_wrapper
        self.epsilon = epsilon
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.img_size = 224  # Model input size
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
    def preprocess_image(self, image_path):
        """
        Preprocess image while maintaining original dimensions for later
        """
        # Read original image and store its size
        original_image = Image.open(image_path).convert('RGB')
        self.original_size = original_image.size  # Store original size
        
        # Create transform pipeline for model input
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transform(original_image).unsqueeze(0).to(self.device)

    def save_perturbed_image(self, perturbed_tensor, save_path):
        """
        Save perturbed image while maintaining original quality and dimensions
        """
        # Denormalize
        mean = torch.tensor(self.mean).view(3, 1, 1).to(self.device)
        std = torch.tensor(self.std).view(3, 1, 1).to(self.device)
        perturbed_tensor = perturbed_tensor * std + mean
        
        # Ensure values are in valid range [0, 1]
        perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
        
        # Convert to PIL Image
        perturbed_image = transforms.ToPILImage()(perturbed_tensor.squeeze(0).cpu())
        
        # Resize back to original dimensions using high-quality interpolation
        if hasattr(self, 'original_size'):
            perturbed_image = perturbed_image.resize(
                self.original_size, 
                Image.Resampling.LANCZOS  # High-quality resampling
            )
        
        # Save with high quality
        perturbed_image.save(
            save_path, 
            'PNG',  # Use PNG for lossless compression
            quality=100,  # Maximum quality
            optimize=True  # Optimize file size without losing quality
        )

    def untargeted_fgsm_attack(self, image_path, step_size=0.00001):
        """
        FGSM attack with improved image handling
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        image.requires_grad = True
        
        # Get OCR embedding
        try:
            ocr_emb = ocr_main(
                image_path=Image.open(image_path), 
                model=self.wrapper.OCR_MODEL
            )
            ocr_emb = ocr_emb.to(self.device)
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            raise
            
        # Rest of the FGSM attack code remains the same
        try:
            logo_feat = self.wrapper.SIAMESE_MODEL.features(image, ocr_emb)
            logo_feat = F.normalize(logo_feat, p=2, dim=1)
            
            ref_embeddings = torch.tensor(self.wrapper.LOGO_FEATS, device=self.device)
            ref_embeddings = F.normalize(ref_embeddings, p=2, dim=1)
            
            similarities = torch.mm(
                logo_feat.view(logo_feat.size(0), -1),
                ref_embeddings.view(ref_embeddings.size(0), -1).t()
            )
            max_similarity = torch.max(similarities)
            
            loss = -max_similarity
            loss.backward()
            
            # Create perturbation
            perturbation = step_size * image.grad.sign()
            perturbed_image = image - perturbation
            
            # Project perturbation back to epsilon ball and valid image range
            delta = perturbed_image - image
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
            
            return perturbed_image
            
        except Exception as e:
            print(f"Error in FGSM attack: {str(e)}")
            raise

    def untargeted_pgd_attack(self, image_path, num_steps=10, step_size=0.00001, alpha=0.8):
        """
        PGD attack with improved image handling
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        
        # Random initialization within epsilon ball
        delta = torch.zeros_like(image, requires_grad=True)
        delta.data.uniform_(-alpha * self.epsilon, alpha * self.epsilon)
        perturbed_image = torch.clamp(image + delta, 0, 1).detach()
        
        ref_embeddings = torch.tensor(self.wrapper.LOGO_FEATS, device=self.device)
        ref_embeddings = F.normalize(ref_embeddings, p=2, dim=1)
        
        # Convert original image to PIL for OCR reference
        original_pil = transforms.ToPILImage()(image.squeeze())
        
        for step in range(num_steps):
            perturbed_image.requires_grad = True
            
            try:
                # Convert current perturbed image to PIL for OCR
                perturbed_pil = transforms.ToPILImage()(perturbed_image.squeeze())
                
                ocr_emb = ocr_main(
                    image_path=perturbed_pil, 
                    model=self.wrapper.OCR_MODEL
                )
                ocr_emb = ocr_emb.to(self.device)
                
                logo_feat = self.wrapper.SIAMESE_MODEL.features(perturbed_image, ocr_emb)
                logo_feat = F.normalize(logo_feat, p=2, dim=1)
                
                similarities = torch.mm(
                    logo_feat.view(logo_feat.size(0), -1),
                    ref_embeddings.view(ref_embeddings.size(0), -1).t()
                )
                max_similarity = torch.max(similarities)
                
                loss = -max_similarity
                loss.backward()
                
                with torch.no_grad():
                    grad_sign = perturbed_image.grad.sign()
                    perturbed_image = perturbed_image - step_size * grad_sign
                    
                    # Project back to epsilon ball around original image
                    delta = perturbed_image - image
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    perturbed_image = torch.clamp(image + delta, 0, 1).detach()
                    
            except Exception as e:
                print(f"Error in PGD step {step}: {str(e)}")
                raise
                
        return perturbed_image

def calculate_attack_metrics(orig_category, orig_conf, orig_target, 
                          attack_category, attack_conf, attack_target):
    """
    Calculate comprehensive metrics for attack success
    
    Args:
        orig_category: Original phishing detection category (0 or 1)
        orig_conf: Original confidence score
        orig_target: Original target brand/domain
        attack_category: Post-attack phishing detection category
        attack_conf: Post-attack confidence score
        attack_target: Post-attack target brand/domain
        
    Returns:
        dict: Dictionary containing various attack metrics
    """
    metrics = {
        'category_change': {
            'success': False,
            'type': None,
            'description': None
        },
        'confidence_impact': {
            'change': None,
            'percent_decrease': None,
            'significant_decrease': False  # >30% decrease is considered significant
        },
        'target_impact': {
            'changed': False,
            'orig_target': orig_target,
            'new_target': attack_target
        },
        'overall_success': False
    }
    
    # Category Change Analysis
    if orig_category == 1 and attack_category == 0:
        metrics['category_change'].update({
            'success': True,
            'type': 'phish_to_benign',
            'description': 'Successfully converted phishing detection to benign'
        })
    elif orig_category == 0 and attack_category == 1:
        metrics['category_change'].update({
            'success': True,
            'type': 'benign_to_phish',
            'description': 'Successfully converted benign detection to phishing'
        })
    else:
        metrics['category_change'].update({
            'success': False,
            'type': 'no_change',
            'description': 'Failed to change detection category'
        })
    
    # Confidence Impact Analysis
    if orig_conf is not None and attack_conf is not None:
        conf_change = orig_conf - attack_conf
        percent_decrease = (conf_change / orig_conf) * 100 if orig_conf != 0 else 0
        
        metrics['confidence_impact'].update({
            'change': float(conf_change),
            'percent_decrease': float(percent_decrease),
            'significant_decrease': percent_decrease > 30
        })
    
    # Target Brand/Domain Impact
    if orig_target != attack_target:
        metrics['target_impact'].update({
            'changed': True,
            'change_type': 'complete' if attack_target is None else 'different'
        })
    
    # Overall Success Criteria
    metrics['overall_success'] = any([
        metrics['category_change']['success'],
        metrics['confidence_impact'].get('significant_decrease', False),
        metrics['target_impact']['changed']
    ])
    
    return metrics

def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_results(results_with_metrics, output_path):
    """
    Save results with proper type conversion
    """
    try:
        serializable_results = convert_to_serializable(results_with_metrics)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        try:
            simple_results = {
                'error': 'Failed to save full results',
                'basic_metrics': {
                    'total_samples': results_with_metrics.get('metrics_summary', {}).get('total_samples', 0),
                    'successful_attacks': {
                        k: v.get('count', 0) 
                        for k, v in results_with_metrics.get('metrics_summary', {}).get('successful_attacks', {}).items()
                    }
                }
            }
            with open(output_path, 'w') as f:
                json.dump(simple_results, f, indent=2)
        except:
            print("Failed to save even basic results")

def calculate_attack_metrics(orig_category, orig_conf, orig_target, 
                          attack_category, attack_conf, attack_target):
    """
    Calculate comprehensive metrics for attack success
    """
    metrics = {
        'category_change': {
            'success': False,
            'type': None,
            'description': None
        },
        'confidence_impact': {
            'change': None,
            'percent_decrease': None,
            'significant_decrease': False
        },
        'target_impact': {
            'changed': False,
            'orig_target': orig_target,
            'new_target': attack_target
        },
        'overall_success': False
    }
    
    # Category Change Analysis
    if orig_category == 1 and attack_category == 0:
        metrics['category_change'].update({
            'success': True,
            'type': 'phish_to_benign',
            'description': 'Successfully converted phishing detection to benign'
        })
    elif orig_category == 0 and attack_category == 1:
        metrics['category_change'].update({
            'success': True,
            'type': 'benign_to_phish',
            'description': 'Successfully converted benign detection to phishing'
        })
    else:
        metrics['category_change'].update({
            'success': False,
            'type': 'no_change',
            'description': 'Failed to change detection category'
        })
    
    # Confidence Impact Analysis
    if orig_conf is not None and attack_conf is not None:
        conf_change = float(orig_conf) - float(attack_conf)
        percent_decrease = (conf_change / float(orig_conf)) * 100 if float(orig_conf) != 0 else 0
        
        metrics['confidence_impact'].update({
            'change': float(conf_change),
            'percent_decrease': float(percent_decrease),
            'significant_decrease': percent_decrease > 30
        })
    
    # Target Brand/Domain Impact
    if orig_target != attack_target:
        metrics['target_impact'].update({
            'changed': True,
            'change_type': 'complete' if attack_target is None else 'different'
        })
    
    # Overall Success Criteria
    metrics['overall_success'] = any([
        metrics['category_change']['success'],
        metrics['confidence_impact'].get('significant_decrease', False),
        metrics['target_impact']['changed']
    ])
    
    return metrics



def process_test_directory(input_dir, output_dir, num_samples=10):
    """
    Process randomly sampled test sites from the input directory with comprehensive visualizations
    """
    # Initialize models
    phishintention = PhishIntentionWrapper()
    attacker = PhishIntentionAttack(phishintention)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"attack_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all valid site folders
    all_sites = [f for f in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, f)) and 
                 os.path.exists(os.path.join(input_dir, f, "shot.png")) and
                 os.path.exists(os.path.join(input_dir, f, "info.txt"))]
    
    # Randomly sample sites
    total_sites = len(all_sites)
    num_samples = min(num_samples, total_sites)
    sampled_sites = np.random.choice(all_sites, size=num_samples, replace=False)
    
    print(f"Randomly sampled {num_samples} sites from total {total_sites} sites")

    # Initialize results and metrics
    all_results = {}
    metrics_summary = {
        'total_samples': num_samples,
        'successful_attacks': {
            'fgsm': {'count': 0, 'details': {'category_changes': 0, 'confidence_impacts': 0, 'target_changes': 0}},
            'pgd': {'count': 0, 'details': {'category_changes': 0, 'confidence_impacts': 0, 'target_changes': 0}}
        },
        'attack_success_rates': {},
        'confidence_impact_stats': {
            'fgsm': {'mean': 0.0, 'max': 0.0},
            'pgd': {'mean': 0.0, 'max': 0.0}
        }
    }

    # Save sampled sites list
    with open(os.path.join(output_dir, 'sampled_sites.txt'), 'w') as f:
        for site in sampled_sites:
            f.write(f"{site}\n")

    # Process each sampled site
    for site_folder in tqdm(sampled_sites, desc="Processing sites", position=0):
        site_path = os.path.join(input_dir, site_folder)
        print(f"\nProcessing site: {site_folder}")

        screenshot_path = os.path.join(site_path, "shot.png")
        info_path = os.path.join(site_path, "info.txt")

        with open(info_path, 'r') as f:
            url = f.read().strip()
        print(f"URL: {url}")

        site_output_dir = os.path.join(output_dir, site_folder)
        os.makedirs(site_output_dir, exist_ok=True)

        try:
            # Get original prediction
            orig_category, orig_target, orig_domain, orig_vis, orig_conf, _, _, _ = \
                phishintention.test_orig_phishintention(url, screenshot_path)
            
            print(f"Original prediction: category={orig_category}, target={orig_target}")

            site_results = {
                'url': url,
                'original': {
                    'category': int(orig_category),
                    'target': orig_target,
                    'domain': orig_domain,
                    'confidence': float(orig_conf) if orig_conf else None
                },
                'attacks': {}
            }

            if orig_vis is not None:
                cv2.imwrite(os.path.join(site_output_dir, "original_vis.png"), orig_vis)

        except Exception as e:
            print(f"Error in original prediction: {str(e)}")
            continue

        # Perform attacks with metrics tracking
        for attack_type in tqdm(['fgsm', 'pgd'], desc="Performing attacks", position=1, leave=False):
            print(f"\nPerforming {attack_type.upper()} attack...")
            try:
                perturbed_path = os.path.join(site_output_dir, f"{attack_type}_perturbed.png")
                
                if attack_type == 'fgsm':
                    perturbed = attacker.untargeted_fgsm_attack(screenshot_path)
                else:
                    perturbed = attacker.untargeted_pgd_attack(screenshot_path)
                    
                attacker.save_perturbed_image(perturbed, perturbed_path)
                
                # Test perturbed image
                results = phishintention.test_orig_phishintention(url, perturbed_path)
                phish_category, pred_target, matched_domain, plotvis, conf = results[:5]
                
                print(f"Attack results - category: {phish_category}, target: {pred_target}")
                
                if plotvis is not None:
                    cv2.imwrite(os.path.join(site_output_dir, f"{attack_type}_vis.png"), plotvis)

                # Generate pixel difference heatmap for this attack
                try:
                    orig_vis_path = os.path.join(site_output_dir, "original_vis.png")
                    plot_pixel_difference_heatmap(
                        orig_vis_path,
                        perturbed_path,
                        os.path.join(site_output_dir, f"{attack_type}_difference_heatmap.png")
                    )
                except Exception as e:
                    print(f"Error generating pixel difference heatmap for {attack_type}: {str(e)}")
                
                # Calculate attack metrics
                attack_metrics = calculate_attack_metrics(
                    orig_category=orig_category,
                    orig_conf=orig_conf,
                    orig_target=orig_target,
                    attack_category=phish_category,
                    attack_conf=conf,
                    attack_target=pred_target
                )
                
                site_results['attacks'][attack_type] = {
                    'metrics': attack_metrics,
                    'category': int(phish_category),
                    'predicted_target': pred_target,
                    'matched_domain': matched_domain,
                    'confidence': float(conf) if conf else None,
                    'success': bool(attack_metrics['overall_success'])
                }
                
                # Update metrics summary
                if attack_metrics['overall_success']:
                    metrics_summary['successful_attacks'][attack_type]['count'] += 1
                if attack_metrics['category_change']['success']:
                    metrics_summary['successful_attacks'][attack_type]['details']['category_changes'] += 1
                if attack_metrics['confidence_impact'].get('significant_decrease', False):
                    metrics_summary['successful_attacks'][attack_type]['details']['confidence_impacts'] += 1
                if attack_metrics['target_impact']['changed']:
                    metrics_summary['successful_attacks'][attack_type]['details']['target_changes'] += 1
                    
                # Update confidence impact stats
                if attack_metrics['confidence_impact']['change'] is not None:
                    current_mean = metrics_summary['confidence_impact_stats'][attack_type]['mean']
                    current_count = metrics_summary['successful_attacks'][attack_type]['count']
                    new_value = attack_metrics['confidence_impact']['change']
                    
                    metrics_summary['confidence_impact_stats'][attack_type]['mean'] = \
                        float((current_mean * current_count + new_value) / (current_count + 1))
                    
                    metrics_summary['confidence_impact_stats'][attack_type]['max'] = float(max(
                        metrics_summary['confidence_impact_stats'][attack_type]['max'],
                        abs(new_value)
                    ))
                
            except Exception as e:
                print(f"{attack_type.upper()} attack failed: {str(e)}")
                site_results['attacks'][attack_type] = {'error': str(e)}

        # Generate confidence comparison plot for this site
        try:
            orig_conf = site_results['original']['confidence']
            fgsm_conf = site_results['attacks'].get('fgsm', {}).get('confidence')
            pgd_conf = site_results['attacks'].get('pgd', {}).get('confidence')
            
            plot_confidence_comparison(
                orig_conf if orig_conf else 0,
                fgsm_conf if fgsm_conf else 0,
                pgd_conf if pgd_conf else 0,
                os.path.join(site_output_dir, "confidence_comparison.png")
            )
        except Exception as e:
            print(f"Error generating confidence comparison for site: {str(e)}")

        all_results[site_folder] = site_results

        # Save intermediate results
        results_with_metrics = {
            'metrics_summary': metrics_summary,
            'site_results': all_results
        }
        save_results(results_with_metrics, os.path.join(output_dir, 'results.json'))

    # Calculate final success rates
    for attack_type in ['fgsm', 'pgd']:
        metrics_summary['attack_success_rates'][attack_type] = {
            'overall': float(metrics_summary['successful_attacks'][attack_type]['count']) / num_samples,
            'category_change': float(metrics_summary['successful_attacks'][attack_type]['details']['category_changes']) / num_samples,
            'confidence_impact': float(metrics_summary['successful_attacks'][attack_type]['details']['confidence_impacts']) / num_samples,
            'target_change': float(metrics_summary['successful_attacks'][attack_type]['details']['target_changes']) / num_samples
        }

    # Generate overall visualizations
    try:
        # Plot overall attack success rates
        plot_attack_success_rates(
            metrics_summary,
            os.path.join(output_dir, 'attack_success_rates.png')
        )
        
        # Plot confidence impact distribution
        plot_confidence_impact_distribution(
            all_results,
            os.path.join(output_dir, 'confidence_impact_distribution.png')
        )
    except Exception as e:
        print(f"Error generating overall visualizations: {str(e)}")

    # Generate final report
    report_path = os.path.join(output_dir, 'Whitebox_attack_report.txt')
    with open(report_path, 'w') as f:
        f.write("=== Adversarial Attack Analysis Report ===\n\n")
        f.write(f"Total Samples Processed: {num_samples}\n\n")
        
        for attack_type in ['fgsm', 'pgd']:
            f.write(f"\n{attack_type.upper()} Attack Results:\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total Successful Attacks: {metrics_summary['successful_attacks'][attack_type]['count']}\n")
            f.write(f"Success Rate: {metrics_summary['attack_success_rates'][attack_type]['overall']:.2%}\n\n")
            
            f.write("Success Breakdown:\n")
            f.write(f"- Category Changes: {metrics_summary['successful_attacks'][attack_type]['details']['category_changes']} ")
            f.write(f"({metrics_summary['attack_success_rates'][attack_type]['category_change']:.2%})\n")
            f.write(f"- Significant Confidence Impacts: {metrics_summary['successful_attacks'][attack_type]['details']['confidence_impacts']} ")
            f.write(f"({metrics_summary['attack_success_rates'][attack_type]['confidence_impact']:.2%})\n")
            f.write(f"- Target Brand Changes: {metrics_summary['successful_attacks'][attack_type]['details']['target_changes']} ")
            f.write(f"({metrics_summary['attack_success_rates'][attack_type]['target_change']:.2%})\n\n")
            
            f.write("Confidence Impact Statistics:\n")
            f.write(f"- Mean Impact: {metrics_summary['confidence_impact_stats'][attack_type]['mean']:.4f}\n")
            f.write(f"- Maximum Impact: {metrics_summary['confidence_impact_stats'][attack_type]['max']:.4f}\n\n")

    print(f"\nDetailed attack report saved to: {report_path}")
    return output_dir, results_with_metrics

if __name__ == "__main__":
    # Set input and output directories
    input_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/datasets/test_sites"
    output_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/WhiteBox_attack_results2"
    
    try:
        # Set random seed for reproducibility
        #np.random.seed(42)
        
        print("Starting White Box Attack Analysis...")
        print(f"Input Directory: {input_directory}")
        print(f"Output Directory: {output_directory}")
        
        # Process randomly sampled sites
        output_dir, results = process_test_directory(
            input_directory, 
            output_directory, 
            num_samples=150
        )
        
        # Print final summary with error handling
        print("\nFinal Attack Summary:")
        metrics_summary = results.get('metrics_summary', {})
        
        if not metrics_summary:
            print("Warning: No metrics summary available")
        else:
            print("\nOverall Results:")
            print("-" * 50)
            
            for attack_type in ['fgsm', 'pgd']:
                try:
                    attack_rates = metrics_summary['attack_success_rates'].get(attack_type, {})
                    attack_stats = metrics_summary['confidence_impact_stats'].get(attack_type, {})
                    
                    print(f"\n{attack_type.upper()} Attack Results:")
                    print("=" * 30)
                    
                    # Overall success rates
                    print(f"Total Success Rate: {attack_rates.get('overall', 0):.2%}")
                    
                    # Detailed breakdown
                    print("\nSuccess Breakdown:")
                    print(f"- Category Changes: {attack_rates.get('category_change', 0):.2%}")
                    print(f"- Confidence Impacts: {attack_rates.get('confidence_impact', 0):.2%}")
                    print(f"- Target Changes: {attack_rates.get('target_change', 0):.2%}")
                    
                    # Confidence impact statistics
                    print("\nConfidence Impact:")
                    print(f"- Mean Impact: {attack_stats.get('mean', 0):.4f}")
                    print(f"- Maximum Impact: {attack_stats.get('max', 0):.4f}")
                    
                except KeyError as e:
                    print(f"Warning: Missing data for {attack_type} attack - {str(e)}")
                except Exception as e:
                    print(f"Error processing results for {attack_type} attack: {str(e)}")
            
            # Print success counts
            print("\nAbsolute Success Counts:")
            print("-" * 50)
            for attack_type in ['fgsm', 'pgd']:
                try:
                    success_counts = metrics_summary['successful_attacks'][attack_type]
                    print(f"\n{attack_type.upper()}:")
                    print(f"Total Successful Attacks: {success_counts.get('count', 0)}")
                    
                    details = success_counts.get('details', {})
                    print(f"- Category Changes: {details.get('category_changes', 0)}")
                    print(f"- Confidence Impacts: {details.get('confidence_impacts', 0)}")
                    print(f"- Target Changes: {details.get('target_changes', 0)}")
                except Exception as e:
                    print(f"Error processing success counts for {attack_type}: {str(e)}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        print("\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: Required directory or file not found - {str(e)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
    finally:
        print("\nExecution finished.")
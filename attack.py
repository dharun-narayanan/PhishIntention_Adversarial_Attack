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

# Import PhishIntention components
from phishintention import PhishIntentionWrapper
from configs import load_config

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
from modules.logo_matching import ocr_main

class PhishIntentionAttack:
    def __init__(self, phishintention_wrapper, epsilon=0.1):
        self.wrapper = phishintention_wrapper
        self.epsilon = epsilon
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.img_size = 224
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image).unsqueeze(0).to(self.device)

    def save_perturbed_image(self, perturbed_tensor, save_path):
        mean = torch.tensor(self.mean).view(3, 1, 1).to(self.device)
        std = torch.tensor(self.std).view(3, 1, 1).to(self.device)
        perturbed_tensor = perturbed_tensor * std + mean
        perturbed_image = transforms.ToPILImage()(perturbed_tensor.squeeze(0).cpu())
        perturbed_image.save(save_path)

    def untargeted_fgsm_attack(self, image_path, step_size=0.01):
        """
        Perform untargeted FGSM attack to try to avoid detection
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        image.requires_grad = True
        
        print("Image tensor shape:", image.shape)
        
        # Get OCR embedding using ocr_main
        try:
            ocr_emb = ocr_main(image_path=Image.open(image_path), 
                              model=self.wrapper.OCR_MODEL)
            print("OCR embedding shape:", ocr_emb.shape)
            ocr_emb = ocr_emb.to(self.device)
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            raise
            
        # Forward pass through the model
        try:
            logo_feat = self.wrapper.SIAMESE_MODEL.features(image, ocr_emb)
            print("Logo features shape:", logo_feat.shape)
            logo_feat = F.normalize(logo_feat, p=2, dim=1)
        except Exception as e:
            print(f"Error in Siamese network processing: {str(e)}")
            raise
        
        # Process reference embeddings
        try:
            ref_embeddings = torch.tensor(self.wrapper.LOGO_FEATS, device=self.device)
            print("Reference embeddings shape:", ref_embeddings.shape)
            ref_embeddings = F.normalize(ref_embeddings, p=2, dim=1)
            
            # Compute similarities
            similarities = torch.mm(logo_feat.view(logo_feat.size(0), -1), 
                                 ref_embeddings.view(ref_embeddings.size(0), -1).t())
            print("Similarities shape:", similarities.shape)
            max_similarity = torch.max(similarities)
        except Exception as e:
            print(f"Error in similarity computation: {str(e)}")
            raise
        
        # Compute gradient
        loss = -max_similarity
        loss.backward()
        
        # Create perturbation
        perturbed_image = image - self.epsilon * image.grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image

    def untargeted_pgd_attack(self, image_path, num_steps=10, step_size=0.01):
        """
        Perform untargeted PGD attack to try to avoid detection
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        perturbed_image = image.clone()
        
        print("Image tensor shape:", image.shape)
        
        # Get reference embeddings once
        ref_embeddings = torch.tensor(self.wrapper.LOGO_FEATS, device=self.device)
        print("Reference embeddings shape:", ref_embeddings.shape)
        ref_embeddings = F.normalize(ref_embeddings, p=2, dim=1)
        
        for step in range(num_steps):
            perturbed_image.requires_grad = True
            
            try:
                # Get OCR embedding using ocr_main
                ocr_emb = ocr_main(image_path=Image.open(image_path), 
                                 model=self.wrapper.OCR_MODEL)
                print(f"Step {step} - OCR embedding shape:", ocr_emb.shape)
                ocr_emb = ocr_emb.to(self.device)
                
                # Forward pass
                logo_feat = self.wrapper.SIAMESE_MODEL.features(perturbed_image, ocr_emb)
                print(f"Step {step} - Logo features shape:", logo_feat.shape)
                logo_feat = F.normalize(logo_feat, p=2, dim=1)
                
                # Compute similarities
                similarities = torch.mm(logo_feat.view(logo_feat.size(0), -1),
                                     ref_embeddings.view(ref_embeddings.size(0), -1).t())
                print(f"Step {step} - Similarities shape:", similarities.shape)
                max_similarity = torch.max(similarities)
                
                # Compute gradient
                loss = -max_similarity
                loss.backward()
                
                # Update image
                with torch.no_grad():
                    perturbation = step_size * perturbed_image.grad.sign()
                    perturbed_image = perturbed_image - perturbation
                    
                    # Project back to epsilon ball
                    delta = perturbed_image - image
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    perturbed_image = torch.clamp(image + delta, 0, 1)
                    
            except Exception as e:
                print(f"Error in PGD step {step}: {str(e)}")
                raise
                
        return perturbed_image

# def process_test_directory(input_dir, output_dir):
#     """
#     Process all test sites in the input directory
#     """
#     # Initialize models
#     phishintention = PhishIntentionWrapper()
#     attacker = PhishIntentionAttack(phishintention)

#     # Create output directory with timestamp
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     output_dir = os.path.join(output_dir, f"attack_results_{timestamp}")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get all folders and sample randomly
#     all_sites = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
#     total_sites = len(all_sites)
#     num_samples = min(num_samples, total_sites)
#     sampled_sites = np.random.choice(all_sites, size=num_samples, replace=False)
    
#     print(f"Randomly sampled {num_samples} sites from total {total_sites} sites")

#     # Results dictionary
#     all_results = {}

#     # Iterate through all folders in input directory
#     for site_folder in tqdm(os.listdir(input_dir), desc="Processing sites"):
#         site_path = os.path.join(input_dir, site_folder)
#         if not os.path.isdir(site_path):
#             continue

#         print(f"\nProcessing site: {site_folder}")

#         # Check for required files
#         screenshot_path = os.path.join(site_path, "shot.png")
#         info_path = os.path.join(site_path, "info.txt")
        
#         if not os.path.exists(screenshot_path) or not os.path.exists(info_path):
#             print(f"Skipping {site_folder}: Missing required files")
#             continue

#         # Read URL from info.txt
#         with open(info_path, 'r') as f:
#             url = f.read().strip()

#         print(f"URL: {url}")

#         # Create output folder for this site
#         site_output_dir = os.path.join(output_dir, site_folder)
#         os.makedirs(site_output_dir, exist_ok=True)

#         # Get original prediction
#         try:
#             orig_category, orig_target, orig_domain, orig_vis, orig_conf, _, _, _ = \
#                 phishintention.test_orig_phishintention(url, screenshot_path)
            
#             print(f"Original prediction: category={orig_category}, target={orig_target}")

#             site_results = {
#                 'url': url,
#                 'original': {
#                     'category': orig_category,
#                     'target': orig_target,
#                     'domain': orig_domain,
#                     'confidence': float(orig_conf) if orig_conf else None
#                 },
#                 'attacks': {}
#             }

#             if orig_category == 1 and orig_vis is not None:
#                 cv2.imwrite(os.path.join(site_output_dir, "original_vis.png"), orig_vis)

#         except Exception as e:
#             print(f"Error in original prediction: {str(e)}")
#             continue

#         # Perform attacks
#         for attack_type in ['fgsm', 'pgd']:
#             print(f"\nPerforming {attack_type.upper()} attack...")
#             try:
#                 perturbed_path = os.path.join(site_output_dir, f"{attack_type}_perturbed.png")
                
#                 if attack_type == 'fgsm':
#                     perturbed = attacker.untargeted_fgsm_attack(screenshot_path)
#                 else:
#                     perturbed = attacker.untargeted_pgd_attack(screenshot_path)
                    
#                 attacker.save_perturbed_image(perturbed, perturbed_path)
                
#                 # Test perturbed image
#                 results = phishintention.test_orig_phishintention(url, perturbed_path)
#                 phish_category, pred_target, matched_domain, plotvis, conf = results[:5]
                
#                 print(f"Attack results - category: {phish_category}, target: {pred_target}")
                
#                 #if phish_category == 1 and plotvis is not None:
#                     #cv2.imwrite(os.path.join(site_output_dir, f"{attack_type}_vis.png"), plotvis)
#                 cv2.imwrite(os.path.join(site_output_dir, f"{attack_type}_vis.png"), plotvis)
                
#                 site_results['attacks'][attack_type] = {
#                     'success': (phish_category == 0 if orig_category == 1 else phish_category == 1),
#                     'category': phish_category,
#                     'predicted_target': pred_target,
#                     'matched_domain': matched_domain,
#                     'confidence': float(conf) if conf else None
#                 }
#             except Exception as e:
#                 print(f"{attack_type.upper()} attack failed: {str(e)}")
#                 site_results['attacks'][attack_type] = {'error': str(e)}

#         all_results[site_folder] = site_results

#         # Save results after each site
#         with open(os.path.join(output_dir, 'results.json'), 'w') as f:
#             json.dump(all_results, f, indent=2)

#     print(f"\nAttack results saved to: {output_dir}")
#     return output_dir, all_results
                
def process_test_directory(input_dir, output_dir, num_samples=1000):
    """
    Process randomly sampled test sites from the input directory
    Args:
        input_dir: Directory containing test sites
        output_dir: Directory to save results
        num_samples: Number of sites to randomly sample (default: 1000)
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
    num_samples = min(num_samples, total_sites)  # Ensure we don't try to sample more than available
    sampled_sites = np.random.choice(all_sites, size=num_samples, replace=False)
    
    print(f"Randomly sampled {num_samples} sites from total {total_sites} sites")

    # Results dictionary
    all_results = {}

    # Iterate through sampled sites
    for site_folder in tqdm(sampled_sites, desc="Processing sites"):
        site_path = os.path.join(input_dir, site_folder)

        print(f"\nProcessing site: {site_folder}")

        # Check for required files (redundant but keeping for safety)
        screenshot_path = os.path.join(site_path, "shot.png")
        info_path = os.path.join(site_path, "info.txt")

        # Read URL from info.txt
        with open(info_path, 'r') as f:
            url = f.read().strip()

        print(f"URL: {url}")

        # Create output folder for this site
        site_output_dir = os.path.join(output_dir, site_folder)
        os.makedirs(site_output_dir, exist_ok=True)

         # Get original prediction
        try:
            orig_category, orig_target, orig_domain, orig_vis, orig_conf, _, _, _ = \
                phishintention.test_orig_phishintention(url, screenshot_path)
            
            print(f"Original prediction: category={orig_category}, target={orig_target}")

            site_results = {
                'url': url,
                'original': {
                    'category': orig_category,
                    'target': orig_target,
                    'domain': orig_domain,
                    'confidence': float(orig_conf) if orig_conf else None
                },
                'attacks': {}
            }

            if orig_category == 1 and orig_vis is not None:
                cv2.imwrite(os.path.join(site_output_dir, "original_vis.png"), orig_vis)

        except Exception as e:
            print(f"Error in original prediction: {str(e)}")
            continue

        # Perform attacks
        for attack_type in ['fgsm', 'pgd']:
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
                
                #if phish_category == 1 and plotvis is not None:
                    #cv2.imwrite(os.path.join(site_output_dir, f"{attack_type}_vis.png"), plotvis)
                cv2.imwrite(os.path.join(site_output_dir, f"{attack_type}_vis.png"), plotvis)
                
                site_results['attacks'][attack_type] = {
                    'success': (phish_category == 0 if orig_category == 1 else phish_category == 1),
                    'category': phish_category,
                    'predicted_target': pred_target,
                    'matched_domain': matched_domain,
                    'confidence': float(conf) if conf else None
                }
            except Exception as e:
                print(f"{attack_type.upper()} attack failed: {str(e)}")
                site_results['attacks'][attack_type] = {'error': str(e)}

        all_results[site_folder] = site_results

        # Save results after each site
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\nAttack results saved to: {output_dir}")
    return output_dir, all_results

if __name__ == "__main__":
    input_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/datasets/test_sites"
    output_directory = "/Users/lakshmid/Documents/Empirical_Security/PhishIntention/WhiteBox_attack_results"
    
    # Set random seed for reproducibility
    #np.random.seed(42)
    
    # Process 1000 randomly sampled sites
    output_dir, results = process_test_directory(input_directory, output_directory, num_samples=100)
    
    # Print summary with robust error handling
    print("\nAttack Summary:")
    for site, site_results in results.items():
        print(f"\nSite: {site}")
        original_target = site_results.get('original', {}).get('target', 'Unknown')
        original_category = site_results.get('original', {}).get('category', 'Unknown')
        print(f"Original: category={original_category}, target={original_target}")
        
        attacks = site_results.get('attacks', {})
        for attack_type, attack_results in attacks.items():
            if attack_results is None or 'error' in attack_results:
                error_msg = attack_results.get('error', 'Unknown error') if attack_results else 'Attack failed'
                print(f"  {attack_type.upper()}: Failed - {error_msg}")
            else:
                try:
                    success = attack_results.get('success', False)
                    category = attack_results.get('category', 'Unknown')
                    confidence = attack_results.get('confidence', 0.0)
                    if confidence is not None:
                        print(f"  {attack_type.upper()}: Success={success}, "
                              f"Category={category}, "
                              f"Confidence={confidence:.4f}")
                    else:
                        print(f"  {attack_type.upper()}: Success={success}, "
                              f"Category={category}, "
                              f"Confidence=N/A")
                except Exception as e:
                    print(f"  {attack_type.upper()}: Error printing results - {str(e)}")
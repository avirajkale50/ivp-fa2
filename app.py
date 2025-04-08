import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import time
from datetime import datetime

class HaarProcessor:
    """Class for Haar transform-based image processing"""
    
    def __init__(self):
        self.wavelet = 'haar'  # Using Haar wavelet
    
    def decompose(self, image, level=1):
        """Apply Haar wavelet decomposition to an image"""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply wavelet transform
        coeffs = pywt.wavedec2(gray, self.wavelet, level=level)
        
        return coeffs
    
    def reconstruct(self, coeffs):
        """Reconstruct image from wavelet coefficients"""
        # Apply inverse wavelet transform
        reconstructed = pywt.waverec2(coeffs, self.wavelet)
        
        # Handle size differences (sometimes reconstruction adds an extra pixel)
        original_shape = coeffs[0].shape
        if reconstructed.shape[0] > original_shape[0] or reconstructed.shape[1] > original_shape[1]:
            reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    def visualize_coefficients(self, coeffs, level):
        """Create visual representation of wavelet coefficients"""
        # Get approximation coefficients (LL band)
        approx = coeffs[0]
        
        # Normalize for visualization
        approx_norm = (approx - approx.min()) / (approx.max() - approx.min() + 1e-10) * 255
        
        # Get the shape of the approximation coefficients
        rows, cols = approx.shape
        
        # Create a visual with appropriate size based on level
        visual = np.zeros((rows * 2, cols * 2))
        
        # Place approximation coefficients
        visual[:rows, :cols] = approx_norm
        
        # Place detail coefficients for the first level
        if len(coeffs) > 1:  # Check if we have detail coefficients
            h_coeffs = abs(coeffs[1][0])  # Horizontal details
            v_coeffs = abs(coeffs[1][1])  # Vertical details
            d_coeffs = abs(coeffs[1][2])  # Diagonal details
            
            # Normalize for better visualization
            h_norm = (h_coeffs - h_coeffs.min()) / (h_coeffs.max() - h_coeffs.min() + 1e-10) * 255
            v_norm = (v_coeffs - v_coeffs.min()) / (v_coeffs.max() - v_coeffs.min() + 1e-10) * 255
            d_norm = (d_coeffs - d_coeffs.min()) / (d_coeffs.max() - d_coeffs.min() + 1e-10) * 255
            
            # Place the detail coefficients
            visual[:rows, cols:cols*2] = h_norm
            visual[rows:rows*2, :cols] = v_norm
            visual[rows:rows*2, cols:cols*2] = d_norm
        
        return visual.astype(np.uint8)
    
    def threshold_coefficients(self, coeffs, method="soft", threshold_percent=10):
        """Apply thresholding to wavelet coefficients (e.g., for denoising)"""
        # Create a copy to avoid modifying the original
        new_coeffs = [coeffs[0].copy()]  # Approximation coefficients remain unchanged
        
        # Calculate threshold value as a percentage of maximum detail coefficient
        all_details = []
        for level in range(1, len(coeffs)):
            for direction in range(3):  # Horizontal, Vertical, Diagonal
                all_details.append(abs(coeffs[level][direction]).max())
        
        max_coeff = max(all_details)
        threshold = max_coeff * threshold_percent / 100
        
        # Apply thresholding to detail coefficients
        for level in range(1, len(coeffs)):
            detail_coeffs = []
            for direction in range(3):  # Horizontal, Vertical, Diagonal
                if method == "hard":
                    # Hard thresholding: set coefficients below threshold to zero
                    detail = coeffs[level][direction].copy()
                    detail[abs(detail) < threshold] = 0
                else:
                    # Soft thresholding: shrink coefficients above threshold
                    detail = coeffs[level][direction].copy()
                    detail = np.sign(detail) * np.maximum(abs(detail) - threshold, 0)
                
                detail_coeffs.append(detail)
            
            new_coeffs.append(tuple(detail_coeffs))
        
        return new_coeffs
    
    def enhance_edges(self, coeffs, factor=2.0):
        """Enhance edges by scaling detail coefficients"""
        # Create a copy to avoid modifying the original
        new_coeffs = [coeffs[0].copy()]  # Approximation coefficients remain unchanged
        
        # Scale detail coefficients
        for level in range(1, len(coeffs)):
            detail_coeffs = []
            for direction in range(3):  # Horizontal, Vertical, Diagonal
                detail = coeffs[level][direction].copy() * factor
                detail_coeffs.append(detail)
            
            new_coeffs.append(tuple(detail_coeffs))
        
        return new_coeffs
    
    def compress(self, coeffs, keep_percent=10):
        """Compress image by keeping only the largest coefficients"""
        # Create a copy to avoid modifying the original
        new_coeffs = [coeffs[0].copy()]  # Keep all approximation coefficients
        
        # For each detail level and direction
        for level in range(1, len(coeffs)):
            detail_coeffs = []
            for direction in range(3):  # Horizontal, Vertical, Diagonal
                # Flatten the detail coefficients
                detail = coeffs[level][direction].copy()
                flat_detail = detail.flatten()
                
                # Sort coefficients by absolute value
                indices = np.argsort(np.abs(flat_detail))
                
                # Calculate how many coefficients to keep
                keep_count = int(len(indices) * keep_percent / 100)
                
                # Zero out the smallest coefficients
                flat_detail[indices[:-keep_count]] = 0
                
                # Reshape back to original size
                detail = flat_detail.reshape(detail.shape)
                detail_coeffs.append(detail)
            
            new_coeffs.append(tuple(detail_coeffs))
        
        return new_coeffs
    
    def process_image(self, image, operation_type, params=None):
        """Apply Haar transform-based processing to an image"""
        if params is None:
            params = {}
        
        # Get decomposition level
        level = params.get('level', 2)
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Store original shape
        original_shape = gray.shape
        
        # Apply wavelet decomposition
        coeffs = self.decompose(gray, level)
        
        # Create visualization of coefficients
        coeff_visual = self.visualize_coefficients(coeffs, level)
        
        # Apply operations using switch-case like approach
        modified_coeffs, operation_name = self.apply_operation(operation_type, coeffs, params)
        
        # Reconstruct image
        result = self.reconstruct(modified_coeffs)
        
        # Ensure result has the same shape as original image
        if result.shape != original_shape:
            result = cv2.resize(result, (original_shape[1], original_shape[0]))
            print(f"Resized reconstructed image from {result.shape} to {original_shape}")
        
        # Calculate metrics
        mse = np.mean((gray.astype(float) - result.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255.0 * 255.0 / mse)
        
        return result, coeff_visual, psnr, mse, operation_name
    
    def apply_operation(self, operation_type, coeffs, params):
        """Apply the specified operation using a switch-case like approach"""
        operations = {
            "denoise": self._apply_denoise,
            "edge_enhance": self._apply_edge_enhance,
            "compress": self._apply_compress
        }
        
        # Default operation if not found
        default_op = lambda c, p: (
            self.threshold_coefficients(c, 'soft', 10),
            "Default Processing"
        )
        
        # Get the operation function or default
        operation_func = operations.get(operation_type, default_op)
        
        # Apply the operation
        return operation_func(coeffs, params)

    def _apply_denoise(self, coeffs, params):
        threshold_percent = params.get('threshold', 20)
        method = params.get('method', 'soft')
        modified_coeffs = self.threshold_coefficients(coeffs, method, threshold_percent)
        operation_name = f"Denoising ({method}, threshold={threshold_percent}%)"
        return modified_coeffs, operation_name

    def _apply_edge_enhance(self, coeffs, params):
        factor = params.get('factor', 2.0)
        modified_coeffs = self.enhance_edges(coeffs, factor)
        operation_name = f"Edge Enhancement (factor={factor})"
        return modified_coeffs, operation_name

    def _apply_compress(self, coeffs, params):
        keep_percent = params.get('keep_percent', 10)
        modified_coeffs = self.compress(coeffs, keep_percent)
        operation_name = f"Compression (keep={keep_percent}%)"
        return modified_coeffs, operation_name

    def visualize_results(self, original, coeffs_visual, processed, psnr, mse, operation_name, save_path=None):
        """Create a visualization of Haar transform processing results"""
        plt.figure(figsize=(18, 10))
        
        # Original image
        plt.subplot(231)
        if len(original.shape) > 2:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Wavelet coefficients visualization
        plt.subplot(232)
        plt.imshow(coeffs_visual, cmap='jet')
        plt.title('Wavelet Coefficients')
        plt.axis('off')
        
        # Processed image
        plt.subplot(233)
        plt.imshow(processed, cmap='gray')
        plt.title(f'Processed Image: {operation_name}')
        plt.axis('off')
        
        # Difference image
        plt.subplot(234)
        if len(original.shape) > 2:
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray_original = original
        diff = gray_original.astype(float) - processed.astype(float)
        plt.imshow(diff, cmap='jet')
        plt.title(f'Difference\nMSE: {mse:.2f}, PSNR: {psnr:.2f}dB')
        plt.axis('off')
        
        # Histogram of original
        plt.subplot(235)
        if len(original.shape) > 2:
            plt.hist(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).flatten(), bins=50, alpha=0.7, color='blue', label='Original')
        else:
            plt.hist(original.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
        plt.title('Histogram Comparison')
        plt.legend()
        
        # Histogram of processed (overlaid)
        plt.subplot(236)
        plt.hist(processed.flatten(), bins=50, alpha=0.7, color='red', label='Processed')
        plt.title('Processed Histogram')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the visualization
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
            
            # Also save the processed image separately
            processed_path = save_path.replace('.png', '_processed.png')
            cv2.imwrite(processed_path, processed)
            print(f"Processed image saved to {processed_path}")
        
        # Show the plot
        plt.show()
        
        # Also show the processed image in a separate window
        cv2.imshow('Processed Image', processed)
        cv2.waitKey(1)  # Wait for key press


class MaskProcessor:
    """Class for video mask processing methods"""
    
    def __init__(self):
        # Background subtraction methods
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.bg_image = None
    
    def background_subtraction(self, frame, method="MOG2"):
        """Generate mask using background subtraction"""
        if method == "MOG2":
            # MOG2 algorithm
            mask = self.bg_subtractor.apply(frame)
            return mask
        
        elif method == "frame_diff":
            # Simple frame differencing
            if self.bg_image is None:
                self.bg_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return np.zeros_like(self.bg_image)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, self.bg_image)
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Update background with current frame (slow adaptation)
            self.bg_image = cv2.addWeighted(self.bg_image, 0.95, gray, 0.05, 0)
            
            return mask
        
        else:
            # Default to MOG2
            return self.bg_subtractor.apply(frame)
    
    def color_segmentation(self, frame, color_range):
        """Generate mask using color-based segmentation"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for specified color range
        lower_bound = np.array(color_range['lower'])
        upper_bound = np.array(color_range['upper'])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        return mask
    
    def apply_morphology(self, mask, operations=None):
        """Apply morphological operations to refine mask"""
        if operations is None:
            # Default operations
            operations = [
                ("erode", 5, 1),  # (operation, kernel_size, iterations)
                ("dilate", 5, 2),
                ("open", 3, 1),
                ("close", 3, 1)
            ]
        
        refined_mask = mask.copy()
        
        for op, kernel_size, iterations in operations:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if op == "erode":
                refined_mask = cv2.erode(refined_mask, kernel, iterations=iterations)
            elif op == "dilate":
                refined_mask = cv2.dilate(refined_mask, kernel, iterations=iterations)
            elif op == "open":
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif op == "close":
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return refined_mask
    
    def detect_contours(self, mask):
        """Detect contours in mask and return frame with drawn contours"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create contour image
        contour_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Draw all contours
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Filter contours by area and draw bounding boxes
        min_area = 100  # Minimum contour area
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Add label
                cv2.putText(contour_image, f"Obj {i+1}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return contour_image, contours
    
    def apply_mask(self, frame, mask):
        """Apply mask to frame to extract objects"""
        # Convert binary mask to 3-channel
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask
        result = cv2.bitwise_and(frame, mask_3channel)
        
        return result
    
    def process_video(self, video_path, mask_method, params=None):
        """Process video using specified mask method"""
        if params is None:
            params = {}
        
        # New parameter to limit number of frames processed
        max_frames = params.get('max_frames', float('inf'))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_frames = []
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_{mask_method}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if specified
            if params.get('skip_frames', 0) > 0:
                if frame_idx % params.get('skip_frames') != 0:
                    frame_idx += 1
                    continue
            
            # Generate mask based on selected method
            if mask_method == "background":
                mask = self.background_subtraction(frame, params.get('bg_method', 'MOG2'))
            
            elif mask_method == "color":
                color_range = params.get('color_range', {
                    'lower': [100, 50, 50],  # Default: blue in HSV
                    'upper': [140, 255, 255]
                })
                mask = self.color_segmentation(frame, color_range)
            
            elif mask_method == "combined":
                # Combine background and color methods
                bg_mask = self.background_subtraction(frame)
                
                # Default to blue color range if not specified
                color_range = params.get('color_range', {
                    'lower': [100, 50, 50],
                    'upper': [140, 255, 255]
                })
                color_mask = self.color_segmentation(frame, color_range)
                
                # Combine masks
                mask = cv2.bitwise_or(bg_mask, color_mask)
            
            else:
                # Default to background subtraction
                mask = self.background_subtraction(frame)
            
            # Apply morphological operations
            morphology_ops = params.get('morphology', [("open", 5, 1), ("close", 5, 1)])
            refined_mask = self.apply_morphology(mask, morphology_ops)
            
            # Detect contours
            contour_image, contours = self.detect_contours(refined_mask)
            
            # Apply mask to original frame
            masked_frame = self.apply_mask(frame, refined_mask)
            
            # Create visualization
            original_resized = cv2.resize(frame, (320, 240))
            mask_resized = cv2.resize(cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR), (320, 240))
            contour_resized = cv2.resize(contour_image, (320, 240))
            masked_resized = cv2.resize(masked_frame, (320, 240))
            
            top_row = np.hstack((original_resized, mask_resized))
            bottom_row = np.hstack((contour_resized, masked_resized))
            result = np.vstack((top_row, bottom_row))
            
            # Add labels
            cv2.putText(result, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, "Mask", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, "Contours", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, "Masked Result", (330, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add frame number
            cv2.putText(result, f"Frame: {frame_idx}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(output_path, result)
            
            output_frames.append(result)
            
            # Display result if requested
            if params.get('display', False):
                cv2.imshow('Video Processing', result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            frame_idx += 1
            processed_count += 1
            
            # Print progress
            if frame_idx % 10 == 0:
                progress = (frame_idx / min(frame_count, max_frames)) * 100 if frame_count > 0 else 0
                print(f"Processing: {progress:.1f}% complete ({processed_count}/{min(frame_count, max_frames)})")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Create video from frames if requested
        if params.get('create_video', True) and output_frames:
            # Define the codec and create VideoWriter object
            output_video_path = os.path.join(output_dir, f"output_{mask_method}.mp4")
            height, width, _ = output_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            for frame in output_frames:
                out.write(frame)
            
            out.release()
            print(f"Video saved to {output_video_path}")
        
        return output_dir


def main():
    """Main function to demonstrate usage of HaarProcessor and MaskProcessor"""
    # Create output directory for results
    output_dir = f"output_haar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Process a single image with Haar transform
    print("Example 1: Image processing with Haar wavelet transform")
    image_path = "example_image.jpg"
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            # Generate a smaller test image to avoid dimension issues
            image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            print("Using random test image (256x256) instead")
        
        # Resize image to manageable dimensions if too large
        max_dim = 512
        h, w = image.shape[:2]
        if h > max_dim or w > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * max_dim / h)
            else:
                new_h, new_w = int(h * max_dim / w), max_dim
            image = cv2.resize(image, (new_w, new_h))
            print(f"Resized image to {new_w}x{new_h} for processing")
        
        haar_processor = HaarProcessor()
        
        # Example operations
        print("\n1. Denoising...")
        result_denoise, coeffs_visual, psnr, mse, operation_name = haar_processor.process_image(
            image, "denoise", {"threshold": 20, "method": "soft", "level": 2}
        )
        # Save the visualization
        save_path = os.path.join(output_dir, "denoise_result.png")
        haar_processor.visualize_results(image, coeffs_visual, result_denoise, psnr, mse, operation_name, save_path)
        
        print("\n2. Edge Enhancement...")
        result_edge, coeffs_visual, psnr, mse, operation_name = haar_processor.process_image(
            image, "edge_enhance", {"factor": 2.5, "level": 2}
        )
        # Save the visualization
        save_path = os.path.join(output_dir, "edge_enhance_result.png")
        haar_processor.visualize_results(image, coeffs_visual, result_edge, psnr, mse, operation_name, save_path)
        
        print("\n3. Compression...")
        result_compress, coeffs_visual, psnr, mse, operation_name = haar_processor.process_image(
            image, "compress", {"keep_percent": 5, "level": 2}  # Reduced level from 3 to 2
        )
        # Save the visualization
        save_path = os.path.join(output_dir, "compress_result.png")
        haar_processor.visualize_results(image, coeffs_visual, result_compress, psnr, mse, operation_name, save_path)
    
    except Exception as e:
        print(f"Error in Example 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Process a video with mask processing
    print("\nExample 2: Video processing with mask methods")
    video_path = "test.mp4"
    
    try:
        mask_processor = MaskProcessor()
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found.")
            # Try to use the webcam instead
            video_path = 0  # Use webcam
            print("Attempting to use webcam instead...")
        
        # Check if we can open the video/webcam
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video or webcam. Skipping video processing example.")
            cap.release()
        else:
            cap.release()
            # Continue with processing
            
            # Example: Background subtraction with simpler parameters
            print("\n1. Background Subtraction...")
            output_dir = mask_processor.process_video(
                video_path, 
                "background", 
                {
                    "bg_method": "MOG2",
                    "morphology": [("dilate", 3, 1)],  # Simplified morphology
                    "skip_frames": 5,  # Skip more frames for faster processing
                    "display": True,
                    "create_video": True,
                    "max_frames": 100  # Process only 100 frames max
                }
            )
            print(f"Results saved to {output_dir}")

    except Exception as e:
        print(f"Error in Example 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up any remaining OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
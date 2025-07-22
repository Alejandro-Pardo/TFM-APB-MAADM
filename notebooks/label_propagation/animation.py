"""
Fixed label spreading animation system with better Jupyter integration
"""
import json
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from typing import Dict, List, Optional
from IPython.display import HTML

class LabelAnimator:
    """Enhanced animator for label spreading visualization with better Jupyter support."""
    
    def __init__(self):
        self.colors = {
            'sink': '#FF6B35',      # Orange
            'source': "#07EA38",    # Green  
            'none': "#2577FA",      # Blue
            'unlabeled': "#C1C1BD"  # Grey
        }
        self.node_size = 200
        # Configure matplotlib for better notebook display
        plt.rcParams['animation.embed_limit'] = 50
        
    def create_animation_from_history(self,
                                     service_name: str,
                                     embeddings: np.ndarray,
                                     method_names: List[str],
                                     history_data: Dict,
                                     save_path: Optional[str] = None,
                                     return_html: bool = True) -> animation.FuncAnimation:
        """
        Create animation using actual propagation history from history.json.
        
        Args:
            service_name: Name of the service
            embeddings: Method embeddings array
            method_names: List of method names
            history_data: History data from history.json
            save_path: Optional path to save animation
            return_html: Whether to return HTML display (for Jupyter)
            
        Returns:
            matplotlib FuncAnimation object or HTML display
        """
        print(f"🎬 Creating history-based animation for {service_name}...")
        
        # Create 2D layout
        print("🎯 Computing 2D layout...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(method_names)//4), 
                   random_state=42, max_iter=1000)
        pos_2d = tsne.fit_transform(embeddings)
        
        # Convert history to animation frames
        print("📚 Processing history data...")
        label_frames = self._convert_history_to_frames(method_names, history_data)
        
        # Create animation with better layout
        print("🎨 Creating animation...")
        
        # Close any existing figures to prevent double display
        plt.close('all')
        
        # Create figure with proper spacing and size
        fig = plt.figure(figsize=(18, 9))
        fig.patch.set_facecolor('white')
        
        # Use gridspec for better control over subplot layout
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], 
                             left=0.06, right=0.95, top=0.90, bottom=0.12, 
                             wspace=0.15, hspace=0.1)
        
        ax1 = fig.add_subplot(gs[0])  # Main visualization
        ax2 = fig.add_subplot(gs[1])  # Statistics
        
        # Pre-calculate plot bounds with proper padding
        x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
        y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = x_range * 0.15  # 15% padding
        padding_y = y_range * 0.15
        
        # Fixed axis limits for consistent display
        xlim = (x_min - padding_x, x_max + padding_x)
        ylim = (y_min - padding_y, y_max + padding_y)
        
        def animate(frame_idx):
            """Animation function with improved rendering."""
            if frame_idx >= len(label_frames):
                return []
                
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Rebuild iterations_data for current frame
            iterations_data = []
            for i in range(frame_idx + 1):
                if i < len(label_frames):
                    current_frame_labels = label_frames[i]
                    label_counts = {'sink': 0, 'source': 0, 'none': 0, 'unlabeled': 0}
                    for method_name in method_names:
                        if method_name in current_frame_labels:
                            label = current_frame_labels[method_name]['label']
                            label_counts[label] += 1
                        else:
                            label_counts['unlabeled'] += 1
                    iterations_data.append((i, label_counts.copy()))
            
            # Get current frame data
            current_labels = label_frames[frame_idx]
            iteration_num = frame_idx
            
            # --- Left plot: Main visualization ---
            colors = []
            sizes = []
            edge_colors = []
            
            for method_name in method_names:
                if method_name in current_labels:
                    label = current_labels[method_name]['label']
                    colors.append(self.colors[label])
                    # Make initial labels slightly larger with different edge
                    if current_labels[method_name].get('initial', False):
                        sizes.append(self.node_size * 1.3)
                        edge_colors.append('black')
                    else:
                        sizes.append(self.node_size)
                        edge_colors.append('white')
                else:
                    colors.append(self.colors['unlabeled'])
                    sizes.append(self.node_size * 0.8)
                    edge_colors.append('gray')
            
            scatter = ax1.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=sizes, 
                                alpha=0.8, edgecolors=edge_colors, linewidth=1.2,
                                zorder=2)
            
            # Set consistent limits and styling
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_title(f'{service_name.upper()} Label Propagation - Iteration {iteration_num}', 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel('t-SNE Dimension 1', fontsize=13)
            ax1.set_ylabel('t-SNE Dimension 2', fontsize=13)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend to main plot
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=12, 
                                         markeredgecolor='black', markeredgewidth=1,
                                         label=f'{label.title()} ({sum(1 for m in method_names if m in current_labels and current_labels[m]["label"] == label)})')
                              for label, color in self.colors.items() if label != 'unlabeled']
            
            # Add unlabeled count
            unlabeled_count = sum(1 for m in method_names if m not in current_labels)
            if unlabeled_count > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=self.colors['unlabeled'], 
                                                 markersize=12, markeredgecolor='gray', 
                                                 markeredgewidth=1,
                                                 label=f'Unlabeled ({unlabeled_count})'))
            
            ax1.legend(handles=legend_elements, loc='upper left', 
                      framealpha=0.95, fancybox=True, shadow=True, fontsize=11)
            
            # --- Right plot: Statistics over time ---
            # Plot statistics
            ax2.set_title('Label Evolution', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('Iteration', fontsize=13)
            ax2.set_ylabel('Number of Methods', fontsize=13)
            
            # Plot lines for all iterations up to current
            if len(iterations_data) > 0:
                frames = [d[0] for d in iterations_data]
                
                for label_name, color in self.colors.items():
                    counts = [d[1][label_name] for d in iterations_data]
                    ax2.plot(frames, counts, color=color, marker='o', 
                            linewidth=3, markersize=8, label=label_name.title(),
                            markeredgecolor='white', markeredgewidth=1)
                
                # Styling for right plot
                ax2.legend(loc='center right', fontsize=11, framealpha=0.95)
                ax2.grid(True, alpha=0.4, linestyle='--')
                
                # Set consistent axis limits
                max_methods = len(method_names)
                ax2.set_ylim(-2, max_methods + 10)
                max_frame = max(len(label_frames), 10)
                ax2.set_xlim(-0.5, max_frame + 0.5)
                
                # Add annotations for current values
                if frame_idx < len(iterations_data):
                    current_label_counts = iterations_data[frame_idx][1]
                    for i, (label_name, color) in enumerate(self.colors.items()):
                        current_count = current_label_counts[label_name]
                        if current_count > 0:
                            ax2.annotate(f'{current_count}', 
                                       xy=(iteration_num, current_count),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=9, fontweight='bold',
                                       color=color, alpha=0.8)
            
            return [scatter]
        
        # Create animation with optimized settings
        anim = animation.FuncAnimation(fig, animate, frames=len(label_frames), 
                                     interval=750, repeat=True, blit=False,
                                     cache_frame_data=False)
        
        # Save if requested
        if save_path:
            print(f"💾 Saving animation to {save_path}...")
            try:
                anim.save(save_path, writer='pillow', fps=0.7, dpi=200, savefig_kwargs={'pad_inches': 0.1})
                size_mb = save_path.stat().st_size / (1024 * 1024)
                print(f"✅ Animation saved! Size: {size_mb:.2f} MB")
            
            except Exception as e:
                print(f"⚠️  Error saving animation: {e}")
        
        # Return appropriate format for Jupyter
        if return_html:
            plt.close(fig)  # Close the figure to prevent double display
            # Convert to HTML with controls
            html_anim = HTML(anim.to_jshtml())
            return html_anim
        else:
            return anim
    
    def _convert_history_to_frames(self, method_names: List[str], history_data: Dict) -> List[Dict]:
        """Convert history data to animation frames with better tracking."""
        frames = []
        cumulative_labels = {}
        
        for iteration_data in history_data['iterations']:
            # Add newly labeled methods
            newly_labeled = iteration_data.get('newly_labeled', {})
            for method_name, label_info in newly_labeled.items():
                if method_name in method_names:  # Only include methods we have embeddings for
                    cumulative_labels[method_name] = label_info
            
            # Create frame with all labeled methods so far
            frames.append(cumulative_labels.copy())
        
        return frames
    

def create_animation(data_manager, service_name: str, save_gif: bool = False, display_in_notebook: bool = True):
    """
    Enhanced main function to create animation from your data with better Jupyter support.
    
    Args:
        data_manager: Your DataManager instance
        service_name: Service to animate (e.g., 'ec2', 'iam', 's3')
        save_gif: Whether to save as GIF file
        display_in_notebook: Whether to return HTML for Jupyter display
        
    Returns:
        HTML animation object for Jupyter or matplotlib animation object
    """

    print("🎬 Animation system ready!")

    service_name_lower = service_name.lower()
    
    if service_name_lower not in data_manager.service_methods:
        print(f"❌ Service '{service_name}' not found")
        available = list(data_manager.service_methods.keys())[:10]
        print(f"Available services: {available}")
        return None
    
    # Get data
    methods = list(data_manager.service_methods[service_name_lower])
    
    # Extract embeddings for this service
    service_embeddings = []
    valid_methods = []
    for method in methods:
        key = (service_name_lower, method)
        if key in data_manager.method_embeddings:
            service_embeddings.append(data_manager.method_embeddings[key])
            valid_methods.append(method)
        else:
            print(f"⚠️ No embedding found for {service_name_lower}.{method}")
    
    if not service_embeddings:
        print(f"❌ No embeddings found for service '{service_name}'")
        return None
        
    embeddings = np.array(service_embeddings)
    methods = valid_methods  # Use only methods with embeddings
    
    # Get initial labels
    initial_labels = {}
    for method in methods:
        key = (service_name_lower, method)
        if key in data_manager.method_labels:
            initial_labels[method] = data_manager.method_labels[key]
        
    if len(initial_labels) == 0:
        print("⚠️  No labeled methods found - animation will show clustering only")
    
    # Create animator
    animator = LabelAnimator()
    
    save_path = None
    if save_gif:
        save_path = config.ANIMATIONS_DIR / f"{service_name}_animation.gif"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    history_file = config.HISTORY_FILE
    if history_file.exists():
        print("📚 Loading propagation history...")
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Check if we have history for this service
            if service_name_lower in history_data:
                return animator.create_animation_from_history(
                    service_name=service_name,
                    embeddings=embeddings,
                    method_names=methods,
                    history_data=history_data[service_name_lower],
                    save_path=save_path,
                    return_html=display_in_notebook
                )
            else:
                print(f"⚠️ No propagation history found for {service_name}")
                return None
        except Exception as e:
            print(f"⚠️ Error loading history file: {e}")
            return None
    else:
        print(f"⚠️ History file not found: {history_file}")
        raise FileNotFoundError(f"History file {history_file} does not exist.")

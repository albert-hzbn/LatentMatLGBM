import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import sys
import numpy as np

# Make PyVista optional to avoid graphics issues
PYVISTA_AVAILABLE = True
try:
    import pyvista as pv
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available - 3D visualization will be disabled")

from predictor import load_charge_density, preprocess_data, predict_property

# ----------------------------
# Config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model paths
ENCODER_PATH = os.path.join(BASE_DIR, "model", "6059_encoder_model_128.h5")
DECODER_PATH = os.path.join(BASE_DIR, "model", "6059_decoder_model_128.h5")


class SplashScreen:
    def __init__(self, parent):
        self.parent = parent
        try:
            self.splash = tk.Toplevel()
            self.splash.title("")
            self.splash.configure(bg='#2c3e50')
            
            # Remove window decorations and center
            self.splash.overrideredirect(True)
            width, height = 400, 200
            
            # Safe window positioning
            try:
                screen_width = self.splash.winfo_screenwidth()
                screen_height = self.splash.winfo_screenheight()
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2
            except:
                x, y = 100, 100  # Fallback position
                
            self.splash.geometry(f'{width}x{height}+{x}+{y}')
            
            # Simple, safe layout
            main_frame = tk.Frame(self.splash, bg='#34495e', relief='ridge', bd=2)
            main_frame.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Title - using simple ASCII
            title_label = tk.Label(
                main_frame, 
                text="LatentMatFusion", 
                font=('Arial', 16, 'bold'),
                fg='white', 
                bg='#34495e'
            )
            title_label.pack(pady=(30, 10))
            
            # Subtitle
            subtitle_label = tk.Label(
                main_frame,
                text="Loading application...",
                font=('Arial', 10),
                fg='#bdc3c7',
                bg='#34495e'
            )
            subtitle_label.pack(pady=(0, 20))
            
            # Simple progress bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                main_frame,
                variable=self.progress_var,
                maximum=100,
                length=250,
                mode='determinate'
            )
            self.progress_bar.pack(pady=10)
            
            # Status text
            self.status_label = tk.Label(
                main_frame,
                text="Initializing...",
                font=('Arial', 9),
                fg='white',
                bg='#34495e'
            )
            self.status_label.pack(pady=(5, 20))
            
        except Exception as e:
            print(f"Splash screen error: {e}")
            self.splash = None
        
    def update_progress(self, value, status):
        if self.splash:
            try:
                self.progress_var.set(value)
                self.status_label.config(text=status)
                self.splash.update()
            except:
                pass  # Ignore update errors
        
    def close(self):
        if self.splash:
            try:
                self.splash.destroy()
            except:
                pass


class LatentSpaceTab:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.latent_data = None
        self.original_shape = None
        self.setup_ui()
    
    def setup_ui(self):
        # Main container for latent space tab
        main_frame = tk.Frame(self.parent_frame, bg='#f8f9fa')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f8f9fa')
        header_frame.pack(fill='x', pady=(0, 15))
        
        title_label = tk.Label(
            header_frame, 
            text="Latent Space Operations", 
            font=('Arial', 20, 'bold'),
            fg='#2c3e50', 
            bg='#f8f9fa'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Load latent space representations and export to CHGCAR format",
            font=('Arial', 10),
            fg='#7f8c8d',
            bg='#f8f9fa'
        )
        subtitle_label.pack(pady=(3, 0))
        
        # Main content frame
        content_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=1)
        content_frame.pack(fill='both', expand=True)
        
        # File loading section
        load_section = tk.LabelFrame(
            content_frame, 
            text="  Load Latent Space Data  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        load_section.pack(fill='x', padx=15, pady=(15, 10))
        
        # Load button
        self.load_latent_button = tk.Button(
            load_section, 
            text="Load Latent Space Data (.npy)",
            command=self.load_latent_data,
            font=('Arial', 10, 'bold'),
            bg='#9b59b6',
            fg='white',
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.load_latent_button.pack(pady=8)
        
        # Info display
        info_frame = tk.Frame(load_section, bg='#ecf0f1', relief='solid', bd=1)
        info_frame.pack(fill='x', padx=5, pady=(5, 8))
        
        self.latent_info_label = tk.Label(
            info_frame, 
            text="No latent space data loaded. Please select a .npy file containing latent representations.",
            font=('Arial', 9),
            fg='#7f8c8d',
            bg='#ecf0f1',
            padx=10,
            pady=10,
            justify='left',
            wraplength=650,
            height=3
        )
        self.latent_info_label.pack(fill='x')
        
        # # Shape configuration section
        # shape_section = tk.LabelFrame(
        #     content_frame, 
        #     text="  Output Shape Configuration  ", 
        #     font=('Arial', 11, 'bold'),
        #     fg='#34495e',
        #     bg='white',
        #     padx=10,
        #     pady=5
        # )
        # shape_section.pack(fill='x', padx=15, pady=10)
        
        # # Shape input frame
        # shape_input_frame = tk.Frame(shape_section, bg='white')
        # shape_input_frame.pack(fill='x', pady=8)
        
        # tk.Label(
        #     shape_input_frame,
        #     text="Target shape (x, y, z):",
        #     font=('Arial', 10),
        #     bg='white'
        # ).pack(side='left', padx=(5, 10))
        
        # # Shape entry fields
        # self.shape_x_var = tk.StringVar(value="32")
        # self.shape_y_var = tk.StringVar(value="32")
        # self.shape_z_var = tk.StringVar(value="32")
        
        # tk.Entry(
        #     shape_input_frame,
        #     textvariable=self.shape_x_var,
        #     width=8,
        #     font=('Arial', 10)
        # ).pack(side='left', padx=2)
        
        # tk.Label(shape_input_frame, text="×", bg='white').pack(side='left', padx=2)
        
        # tk.Entry(
        #     shape_input_frame,
        #     textvariable=self.shape_y_var,
        #     width=8,
        #     font=('Arial', 10)
        # ).pack(side='left', padx=2)
        
        # tk.Label(shape_input_frame, text="×", bg='white').pack(side='left', padx=2)
        
        # tk.Entry(
        #     shape_input_frame,
        #     textvariable=self.shape_z_var,
        #     width=8,
        #     font=('Arial', 10)
        # ).pack(side='left', padx=2)
        
        # Export section
        export_section = tk.LabelFrame(
            content_frame, 
            text="  Export to CHGCAR  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        export_section.pack(fill='x', padx=15, pady=10)
        
        self.export_button = tk.Button(
            export_section, 
            text="Export to CHGCAR Format",
            command=self.export_to_chgcar,
            state=tk.DISABLED,
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.export_button.pack(pady=8)
        
        # Progress section
        progress_section = tk.LabelFrame(
            content_frame, 
            text="  Processing Status  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        progress_section.pack(fill='x', padx=15, pady=(10, 15))
        
        self.latent_progress = ttk.Progressbar(
            progress_section, 
            orient="horizontal", 
            length=400, 
            mode="indeterminate"
        )
        self.latent_progress.pack(pady=8, padx=10, fill='x')
    
    def load_latent_data(self):
        file_path = filedialog.askopenfilename(
            title="Select latent space data file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        self.load_latent_button.configure(text="Loading...", state=tk.DISABLED)
        self.latent_progress.start(10)
        
        def _load():
            try:
                # Load the latent data
                self.latent_data = np.load(file_path)
                
                # Update UI on main thread
                self.parent_frame.after(0, self._update_latent_ui_after_load, file_path)
                
            except Exception as e:
                self.parent_frame.after(0, self._handle_latent_load_error, str(e))
        
        threading.Thread(target=_load, daemon=True).start()
    
    def _update_latent_ui_after_load(self, file_path):
        self.latent_progress.stop()
        self.load_latent_button.configure(
            text="Load Latent Space Data (.npy)", 
            state=tk.NORMAL
        )
        
        filename = os.path.basename(file_path)
        info_text = (f"File: {filename}\n"
                    f"Latent shape: {self.latent_data.shape}\n"
                    f"Data type: {self.latent_data.dtype} | Ready for export")
        
        self.latent_info_label.config(
            text=info_text,
            fg='#27ae60',
            bg='#d5f4e6'
        )
        
        self.export_button.config(state=tk.NORMAL)
    
    def _handle_latent_load_error(self, error_msg):
        self.latent_progress.stop()
        self.load_latent_button.configure(
            text="Load Latent Space Data (.npy)", 
            state=tk.NORMAL
        )
        self.export_button.config(state=tk.DISABLED)
        
        self.latent_info_label.config(
            text=f"Error loading file: {error_msg}",
            fg='#e74c3c',
            bg='#fdf2f2'
        )
        messagebox.showerror("File Load Error", error_msg)
    
    def export_to_chgcar(self):
        if self.latent_data is None:
            messagebox.showerror("Error", "No latent data loaded.")
            return
        
        try:
            # Get target shape
            target_x = int(self.shape_x_var.get())
            target_y = int(self.shape_y_var.get())
            target_z = int(self.shape_z_var.get())
            target_shape = (target_x, target_y, target_z)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for shape dimensions.")
            return
        
        # Get save location
        save_path = filedialog.asksaveasfilename(
            title="Save CHGCAR file",
            defaultextension=".CHGCAR",
            filetypes=[("CHGCAR files", "CHGCAR*"), ("All files", "*.*")]
        )
        if not save_path:
            return
        
        self.export_button.configure(text="Exporting...", state=tk.DISABLED)
        self.latent_progress.start(10)
        
        def _export():
            try:
                # Process latent data and convert to charge density format
                charge_density = self._latent_to_charge_density(self.latent_data, target_shape)
                
                # Export to CHGCAR format
                self._write_chgcar(charge_density, save_path)
                
                self.parent_frame.after(0, lambda: self._export_success(save_path, target_shape))
                
            except Exception as e:
                self.parent_frame.after(0, lambda: self._export_error(str(e)))
        
        threading.Thread(target=_export, daemon=True).start()
    
    def _latent_to_charge_density(self, latent_data, target_shape):
        """Convert latent space data to charge density format"""
        # This is a placeholder - you'll need to implement the actual conversion
        # based on your specific latent space representation
        
        if latent_data.ndim == 1:
            # If it's a 1D latent vector, reshape to target shape
            total_elements = target_shape[0] * target_shape[1] * target_shape[2]
            if latent_data.shape[0] == total_elements:
                charge_density = latent_data.reshape(target_shape)
            else:
                # Interpolate or pad/truncate as needed
                if latent_data.shape[0] < total_elements:
                    # Pad with zeros or interpolate
                    padded = np.pad(latent_data, (0, total_elements - latent_data.shape[0]), 'constant')
                    charge_density = padded.reshape(target_shape)
                else:
                    # Truncate
                    charge_density = latent_data[:total_elements].reshape(target_shape)
        elif latent_data.ndim == 3:
            # If it's already 3D, resize to target shape
            from scipy.ndimage import zoom
            zoom_factors = [target_shape[i] / latent_data.shape[i] for i in range(3)]
            charge_density = zoom(latent_data, zoom_factors, order=1)
        else:
            raise ValueError(f"Unsupported latent data shape: {latent_data.shape}")
        
        # Ensure positive values (typical for charge density)
        charge_density = np.abs(charge_density)
        
        return charge_density
    
    def _write_chgcar(self, charge_density, filepath):
        """Write charge density data to CHGCAR format"""
        # This is a simplified CHGCAR writer - you may need to adjust based on your needs
        with open(filepath, 'w') as f:
            # Write header (simplified)
            f.write("Generated from latent space data\n")
            f.write("1.0\n")
            f.write("10.0 0.0 0.0\n")
            f.write("0.0 10.0 0.0\n")
            f.write("0.0 0.0 10.0\n")
            f.write("C\n")
            f.write("1\n")
            f.write("Direct\n")
            f.write("0.0 0.0 0.0\n")
            f.write("\n")
            
            # Write grid dimensions
            f.write(f"{charge_density.shape[0]} {charge_density.shape[1]} {charge_density.shape[2]}\n")
            
            # Write charge density data
            count = 0
            for value in charge_density.flatten():
                f.write(f"{value:.6e} ")
                count += 1
                if count % 5 == 0:  # New line every 5 values
                    f.write("\n")
            
            if count % 5 != 0:
                f.write("\n")
    
    def _export_success(self, save_path, target_shape):
        self.latent_progress.stop()
        self.export_button.configure(text="Export to CHGCAR Format", state=tk.NORMAL)
        
        filename = os.path.basename(save_path)
        messagebox.showinfo(
            "Export Successful", 
            f"Successfully exported to {filename}\nTarget shape: {target_shape}"
        )
    
    def _export_error(self, error_msg):
        self.latent_progress.stop()
        self.export_button.configure(text="Export to CHGCAR Format", state=tk.NORMAL)
        messagebox.showerror("Export Error", f"Export failed: {error_msg}")


class ChargeDensityGUI:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # Hide main window initially
        self.setup_splash_and_init()
        
    def setup_splash_and_init(self):
        # Show splash screen
        splash = SplashScreen(self.root)
        
        # Simulate loading process
        loading_steps = [
            (25, "Loading models..."),
            (50, "Initializing components..."),
            (75, "Setting up interface..."),
            (100, "Ready!")
        ]
        
        for progress, status in loading_steps:
            splash.update_progress(progress, status)
            time.sleep(0.2)  # Reduced sleep time
            
        splash.close()
        self.init_main_window()
        
    def init_main_window(self):
        self.root.deiconify()  # Show main window
        self.root.title("LatentMatFusion - Property Predictor")
        self.root.configure(bg='#f8f9fa')
        
        # Set fixed window size
        self.root.geometry('800x650')
        self.root.resizable(True, True)
        self.root.minsize(750, 600)
        
        # Center the window
        try:
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
        except:
            pass
        
        self.file_path = None
        self.charge_density = None
        self.processed_data = None
        self.method = None
        self.plotter = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Property Prediction (original functionality)
        self.prediction_frame = tk.Frame(self.notebook, bg='#f8f9fa')
        # Tab 1: Property Prediction (original functionality)
        self.prediction_frame = tk.Frame(self.notebook, bg='#f8f9fa')
        self.notebook.add(self.prediction_frame, text="Property Prediction")
        
        # Tab 2: Latent Space Operations
        self.latent_frame = tk.Frame(self.notebook, bg='#f8f9fa')
        self.notebook.add(self.latent_frame, text="Latent Space")
        
        # Setup Tab 1 content (original functionality)
        self.setup_prediction_tab()
        
        # Setup Tab 2 content (latent space operations)
        self.latent_tab = LatentSpaceTab(self.latent_frame)
        
    def setup_prediction_tab(self):
        # Main container
        main_frame = tk.Frame(self.prediction_frame, bg='#f8f9fa')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f8f9fa')
        header_frame.pack(fill='x', pady=(0, 15))
        
        title_label = tk.Label(
            header_frame, 
            text="LatentMatFusion", 
            font=('Arial', 20, 'bold'),
            fg='#2c3e50', 
            bg='#f8f9fa'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Mechanical Properties Prediction from Charge Density",
            font=('Arial', 10),
            fg='#7f8c8d',
            bg='#f8f9fa'
        )
        subtitle_label.pack(pady=(3, 0))
        
        # Main content in a frame with fixed proportions
        content_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=1)
        content_frame.pack(fill='both', expand=True)
        
        # Use grid for better control
        content_frame.grid_rowconfigure(0, weight=0)  # File section - fixed
        content_frame.grid_rowconfigure(1, weight=0)  # Viz section - fixed  
        content_frame.grid_rowconfigure(2, weight=0)  # Predict section - fixed
        content_frame.grid_rowconfigure(3, weight=0)  # Progress section - fixed
        content_frame.grid_columnconfigure(0, weight=1)
        
        # File loading section
        file_section = tk.LabelFrame(
            content_frame, 
            text="  Data Input  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        file_section.grid(row=0, column=0, sticky='ew', padx=15, pady=(15, 8))
        
        # Load button
        self.load_button = tk.Button(
            file_section, 
            text="Load Charge Density File (CHGCAR)",
            command=self.load_file,
            font=('Arial', 10, 'bold'),
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.load_button.pack(pady=8)
        
        # Info display with fixed height
        info_frame = tk.Frame(file_section, bg='#ecf0f1', relief='solid', bd=1)
        info_frame.pack(fill='x', padx=5, pady=(5, 8))
        
        self.info_label = tk.Label(
            info_frame, 
            text="No file loaded. Please select a CHGCAR or NPY file to begin.",
            font=('Arial', 9),
            fg='#7f8c8d',
            bg='#ecf0f1',
            padx=10,
            pady=10,
            justify='left',
            wraplength=650,
            height=4  # Fixed height in lines
        )
        self.info_label.pack(fill='x')
        
        # Visualization section
        if PYVISTA_AVAILABLE:
            viz_section = tk.LabelFrame(
                content_frame, 
                text="  3D Visualization  ", 
                font=('Arial', 11, 'bold'),
                fg='#34495e',
                bg='white',
                padx=10,
                pady=5
            )
            viz_section.grid(row=1, column=0, sticky='ew', padx=15, pady=8)
            
            self.viz_button = tk.Button(
                viz_section, 
                text="Show 3D Visualization",
                command=self.show_pyvista_volume,
                state=tk.DISABLED,
                font=('Arial', 9),
                bg='#9b59b6',
                fg='white',
                relief='flat',
                padx=15,
                pady=6,
                cursor='hand2'
            )
            self.viz_button.pack(pady=8)
        
        # Prediction section
        predict_section = tk.LabelFrame(
            content_frame, 
            text="  AI Prediction  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        predict_section.grid(row=2, column=0, sticky='ew', padx=15, pady=8)
        
        self.predict_button = tk.Button(
            predict_section, 
            text="Predict Mechanical Properties",
            command=self.predict,
            state=tk.DISABLED,
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.predict_button.pack(pady=8)
        
        # Progress section
        progress_section = tk.LabelFrame(
            content_frame, 
            text="  Processing Status  ", 
            font=('Arial', 11, 'bold'),
            fg='#34495e',
            bg='white',
            padx=10,
            pady=5
        )
        progress_section.grid(row=3, column=0, sticky='ew', padx=15, pady=(8, 15))
        
        self.progress = ttk.Progressbar(
            progress_section, 
            orient="horizontal", 
            length=400, 
            mode="indeterminate"
        )
        self.progress.pack(pady=8, padx=10, fill='x')

    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select charge density file",
            filetypes=[("CHGCAR files", "CHGCAR*"),
                       ("All files", "*.*")]
        )
        if not self.file_path:
            return

        # Show loading state
        self.load_button.configure(text="Loading...", state=tk.DISABLED)
        self.progress.start(10)
        
        def _load():
            try:
                # base = os.path.basename(self.file_path)
                # if base.upper().startswith("CHGCAR"):
                #     input_type = "chgcar"
                # else:
                #     input_type = "chgcar" if "CHGCAR" in base.upper() else "npy"

                # Load and preprocess
                self.charge_density, source = load_charge_density(self.file_path)
                self.processed_data, self.method = preprocess_data(self.charge_density)

                # Update UI on main thread
                self.root.after(0, self._update_ui_after_load, source)
                
            except Exception as e:
                self.root.after(0, self._handle_load_error, str(e))
                
        threading.Thread(target=_load, daemon=True).start()
    
    def _update_ui_after_load(self, source):
        self.progress.stop()
        self.load_button.configure(
            text="Load Charge Density File (CHGCAR)", 
            state=tk.NORMAL
        )
        
        # Create compact info text
        filename = os.path.basename(self.file_path)
        info_text = (f"Source: {source} | Shape: {self.charge_density.shape}\n"
                    f"Method: {self.method} | Processed: {self.processed_data.shape}\n"
                    f"Status: Ready for prediction")
        # info_text = (f"File: {filename}\n"
        #             f"Source: {source} | Shape: {self.charge_density.shape}\n"
        #             f"Method: {self.method} | Processed: {self.processed_data.shape}\n"
        #             f"Status: Ready for prediction")
        
        self.info_label.config(
            text=info_text,
            fg='#27ae60',
            bg='#d5f4e6'
        )

        # Enable buttons
        self.predict_button.config(state=tk.NORMAL)
        if PYVISTA_AVAILABLE and hasattr(self, 'viz_button'):
            self.viz_button.config(state=tk.NORMAL)
    
    def _handle_load_error(self, error_msg):
        self.progress.stop()
        self.load_button.configure(
            text="Load Charge Density File (CHGCAR)", 
            state=tk.NORMAL
        )
        self.predict_button.config(state=tk.DISABLED)
        if PYVISTA_AVAILABLE and hasattr(self, 'viz_button'):
            self.viz_button.config(state=tk.DISABLED)
        
        self.info_label.config(
            text=f"Error loading file: {error_msg}",
            fg='#e74c3c',
            bg='#fdf2f2'
        )
        messagebox.showerror("File Load Error", error_msg)

    def show_pyvista_volume(self):
        if not PYVISTA_AVAILABLE:
            messagebox.showwarning("Feature Unavailable", "PyVista is not available.")
            return
            
        if self.charge_density is None:
            messagebox.showerror("Error", "No charge density data loaded.")
            return

        try:
            if self.plotter is not None:
                try:
                    self.plotter.close()
                except:
                    pass
            volume = self.charge_density.astype(float)

            pv.set_plot_theme("default")
            self.plotter = pv.Plotter(
                title="3D Charge Density Visualization",
                window_size=(800, 600),
                off_screen=False
            )
            
            self.plotter.add_volume(volume, cmap="viridis", opacity="sigmoid_6")
            self.plotter.add_axes()
            self.plotter.show_bounds(grid="front", location="outer")
            
            try:
                self.plotter.show(auto_close=False)
            except Exception as e:
                messagebox.showerror("Visualization Error", f"Could not display 3D visualization: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("Visualization Error", f"3D visualization failed: {str(e)}")

    def predict(self):
        if self.processed_data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        # Update UI for prediction state
        self.load_button.config(state=tk.DISABLED)
        self.predict_button.config(text="Predicting...", state=tk.DISABLED)
        self.progress.start(10)

        def _run():
            try:
                # Predict all three properties
                bulk_modulus = predict_property(self.file_path, self.processed_data, "bulk_modulus", ENCODER_PATH)
                shear_modulus = predict_property(self.file_path, self.processed_data, "shear_modulus", ENCODER_PATH)
                youngs_modulus = predict_property(self.file_path, self.processed_data, "youngs_modulus", ENCODER_PATH)
                formation_energy = predict_property(self.file_path, self.processed_data, "formation_energy", ENCODER_PATH)
                debye_temperature = predict_property(self.file_path, self.processed_data, "debye_temperature", ENCODER_PATH)
                results = {
                    'bulk_modulus': bulk_modulus,
                    'shear_modulus': shear_modulus,
                    'youngs_modulus': youngs_modulus,
                    'formation_energy': formation_energy,
                    'debye_temperature': debye_temperature
                }
                
                self.root.after(0, lambda: self._show_prediction_result(results))
            except Exception as e:
                error_msg = str(e)
                self.root.after(
                    0,
                    lambda msg=error_msg: messagebox.showerror(
                        "Prediction Error",
                        f"Prediction failed: {msg}"
                    )
                )

            finally:
                self.root.after(0, self._reset_ui_after_prediction)

        threading.Thread(target=_run, daemon=True).start()
    
    def _show_prediction_result(self, results):
        """Show prediction results in a custom dialog"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction Results - LatentMatFusion")
        result_window.configure(bg='white')
        result_window.geometry('500x520')
        result_window.resizable(False, False)
        
        # Center the result window
        try:
            result_window.update_idletasks()
            x = (result_window.winfo_screenwidth() // 2) - 250
            y = (result_window.winfo_screenheight() // 2) - 175
            result_window.geometry(f'500x520+{x}+{y}')
        except:
            pass
        
        # Make it modal
        result_window.transient(self.root)
        result_window.grab_set()
        
        main_frame = tk.Frame(result_window, bg='white')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_label = tk.Label(
            main_frame,
            text="Prediction Complete!",
            font=('Arial', 18, 'bold'),
            fg='#27ae60',
            bg='white'
        )
        header_label.pack(pady=(10, 5))
        
        # Subheader
        sub_label = tk.Label(
            main_frame,
            text="AI models have analyzed your charge density data",
            font=('Arial', 10),
            fg='#7f8c8d',
            bg='white'
        )
        sub_label.pack(pady=(0, 20))
        
        # Results display frame
        results_frame = tk.Frame(main_frame, bg='#f8f9fa', relief='solid', bd=1)
        results_frame.pack(fill='x', pady=10)
        
        # Title for results
        tk.Label(
            results_frame,
            text="Predicted Mechanical Properties:",
            font=('Arial', 12, 'bold'),
            fg='#2c3e50',
            bg='#f8f9fa'
        ).pack(pady=(15, 10))
        
        # Individual property results
        properties = [
            ('Bulk Modulus', results['bulk_modulus'], 'GPa', '#3498db'),
            ('Shear Modulus', results['shear_modulus'], 'GPa', '#e74c3c'),
            ('Young\'s Modulus', results['youngs_modulus'], 'GPa', '#f39c12'),
            ('Formation Energy', results['formation_energy'], 'eV', '#46a87a'),
            ('Debye Temperature', results['debye_temperature'], 'K', '#893a8b')
        ]

        for prop_name, value, unit, color in properties:
            prop_frame = tk.Frame(results_frame, bg='white', relief='solid', bd=1)
            prop_frame.pack(fill='x', padx=15, pady=5)


            # LEFT: property name
            tk.Label(
                prop_frame,
                text=f"{prop_name}:",
                font=('Arial', 11),
                fg='#2c3e50',
                bg='white',
                anchor='w'
            ).pack(side='left', padx=10, pady=8)

            # RIGHT: value + unit
            tk.Label(
                prop_frame,
                text=f"{value:.2f} {unit}",
                font=('Arial', 14, 'bold'),
                fg=color,
                bg='white',
                anchor='e'
            ).pack(side='right', padx=10, pady=8)
        
        # Add some padding at the bottom of results
        tk.Label(results_frame, text="", bg='#f8f9fa', height=1).pack()
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(pady=(15, 0))
        
        # Close button
        close_btn = tk.Button(
            button_frame,
            text="Close",
            command=result_window.destroy,
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='flat',
            padx=20,
            pady=6,
            cursor='hand2'
        )
        close_btn.pack()
        
        # Update main window info
        filename = os.path.basename(self.file_path)
        info_text = (f"Predictions: Bulk={results['bulk_modulus']:.2f} | Shear={results['shear_modulus']:.2f} | Young's={results['youngs_modulus']:.2f} GPa | Formation Energy={results['formation_energy']:.2f} eV | Debye Temperature={results['debye_temperature']:.2f} K")
        
        # info_text = (f"File: {filename}\n"
        #             f"Shape: {self.charge_density.shape} -> {self.processed_data.shape}\n"
        #             f"Method: {self.method}\n"
        #             f"Predictions: Bulk={results['bulk_modulus']:.2f} | Shear={results['shear_modulus']:.2f} | Young's={results['youngs_modulus']:.2f} GPa")
        
        self.info_label.config(
            text=info_text,
            fg='#27ae60',
            bg='#d5f4e6'
        )

    def _reset_ui_after_prediction(self):
        self.progress.stop()
        self.load_button.config(state=tk.NORMAL)
        self.predict_button.config(text="Predict Mechanical Properties")

        # Only re-enable Predict if valid data is still loaded
        if self.processed_data is not None:
            self.predict_button.config(state=tk.NORMAL)
        else:
            self.predict_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    try:
        # Add scipy import check for the latent space functionality
        try:
            import scipy.ndimage
        except ImportError:
            print("Warning: scipy not available - some latent space features may be limited")
        
        root = tk.Tk()
        app = ChargeDensityGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
import pathlib
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Custom Tkinter frame for original images
class tkFrameOriginalImages(tk.Frame):
    def __init__(self, master, container):
        super().__init__(master)
        self.container = container

        self._create_widgets()

        # Create image display window
        self.image_display_window = ImageDisplayWindow(self.container)
        # self.homeography_window = ImageDisplayWindow(self.container)
        self.image_display_window.toggle_visibility()

    def _create_widgets(self):
        # Create buttons for selecting images and displaying the image window
        initial_image_button = tk.Button(self, text="Select Initial State Image", command=self.select_initial_image)
        initial_image_button.grid(row=1, column=0, padx=10, pady=10)

        final_image_button = tk.Button(self, text="Select Final State Image", command=self.select_final_image)
        final_image_button.grid(row=1, column=1, padx=10, pady=10)

        self.display_button = tk.Button(self, text="Toggle Image Display", command=self.toggle_image_display)
        self.display_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.display_button = tk.Button(self, text="Start Homeography", command=self.start_homeography)
        self.display_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def select_initial_image(self):
        file_path = filedialog.askopenfilename(title="Select Initial State Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.container.initial_path_image = pathlib.Path(file_path)
            self.container.initial_image = cv2.imread(file_path)
            self.image_display_window.update_images()

    def select_final_image(self):
        file_path = filedialog.askopenfilename(title="Select Final State Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.container.final_path_image = pathlib.Path(file_path)
            self.container.final_image = cv2.imread(file_path)
            self.image_display_window.update_images()

    def start_homeography(self):
        file_path = filedialog.askopenfilename(title="Select Final State Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.container.final_image = cv2.imread(file_path)
            self.image_display_window.update_images()

    def toggle_image_display(self):
        self.image_display_window.toggle_visibility()

# Tkinter window class for image display
class ImageDisplayWindow(tk.Toplevel):
    def __init__(self, container):
        super().__init__()
        self.title("Image Display")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.geometry("800x400")

        self.canvas_initial = tk.Canvas(self, width=400, height=400)
        self.canvas_initial.pack(side=tk.LEFT)

        self.canvas_final = tk.Canvas(self, width=400, height=400)
        self.canvas_final.pack(side=tk.RIGHT)

        self.container = container

        self.visible = False

    def toggle_visibility(self):
        self.visible = not self.visible
        if self.visible:
            self.update_images()
            self.deiconify()
        else:
            self.withdraw()

    def update_images(self):
        self._update_initial_image()
        self._update_final_image()
        
    def _update_initial_image(self):
        try:
            initial_img = Image.fromarray(self.container.get_initial_image())
            initial_width, initial_height = initial_img.size
            initial_scale = min(1, 400 / max(initial_width, initial_height))
            scaled_initial_img = initial_img.resize((int(initial_width * initial_scale), int(initial_height * initial_scale)))
            initial_tkimg = ImageTk.PhotoImage(scaled_initial_img)
            self.canvas_initial.create_image(0, 0, anchor="nw", image=initial_tkimg)
            self.canvas_initial.image_reference = initial_tkimg
        
        except AttributeError:
            self.canvas_initial.delete("all")
            self.canvas_initial.image_reference = None
            pass

    def _update_final_image(self):
        try:
            final_img = Image.fromarray(self.container.get_final_image())
            # Calculate scaling factors for width and height

            final_width, final_height = final_img.size

            final_scale = min(1, 400 / max(final_width, final_height))

            # Scale the images
            scaled_final_img = final_img.resize((int(final_width * final_scale), int(final_height * final_scale)))

            # Convert the scaled images to Tkinter PhotoImage

            final_tkimg = ImageTk.PhotoImage(scaled_final_img)

            # Display the images on the canvases
            self.canvas_final.create_image(0, 0, anchor="nw", image=final_tkimg)

            # Store the references to avoid garbage collection
            self.canvas_final.image_reference = final_tkimg
        except AttributeError:
            # Clear the canvases

            self.canvas_final.delete("all")
            self.canvas_final.image_reference = None


    def on_closing(self):
        pass


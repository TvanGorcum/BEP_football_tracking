import tkinter as tk
from tkinter import filedialog

def launch_gui():
    root = tk.Tk()
    root.title("Football Video Analyzer")
    input_path, output_path = tk.StringVar(), tk.StringVar()
    visualize = tk.BooleanVar()

    def select_input():
        input_path.set(filedialog.askopenfilename(filetypes=[("Video/Image Files", "*.mp4 *.jpg *.png")]))

    def select_output():
        output_path.set(filedialog.askdirectory())

    tk.Label(root, text="Input File:").grid(row=0, column=0)
    tk.Button(root, text="Browse", command=select_input).grid(row=0, column=1)
    tk.Label(root, text="Output folder:").grid(row=1, column=0)
    tk.Button(root, text="Browse", command=select_output).grid(row=1, column=1)
    tk.Checkbutton(root, text="Enable Visualization", variable=visualize).grid(row=2, columnspan=2)
    tk.Button(root, text="Start", command=root.quit).grid(row=3, columnspan=2)

    root.mainloop()
    return input_path.get(), output_path.get(), visualize.get()
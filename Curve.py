import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CurveEditor:
    def __init__(self, master, curve_data, key, apply_callback):
        self.master = tk.Toplevel(master)
        self.master.title(f"Edit Curve: {key}")
        self.key = key
        self.original_curve = np.array(curve_data, dtype=np.float32)
        self.curve = self.original_curve.copy()
        self.apply_callback = apply_callback

        fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot(self.curve, marker='o', lw=1)
        self.ax.set_title(f"Click and drag to edit: {key}")
        self.ax.set_xlim(0, len(self.curve)-1)
        self.ax.set_ylim(np.min(self.curve)*1.1, np.max(self.curve)*1.1)
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack()

        fig.canvas.mpl_connect("button_press_event", self.on_click)
        fig.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.selected_idx = None

        self.build_controls()

    def build_controls(self):
        frame = ttk.Frame(self.master)
        frame.pack(pady=10)

        ttk.Button(frame, text="Normalize", command=self.normalize).pack(side='left', padx=5)
        ttk.Button(frame, text="Smooth", command=self.smooth).pack(side='left', padx=5)
        ttk.Button(frame, text="Quantize", command=lambda: self.quantize(20)).pack(side='left', padx=5)
        ttk.Button(frame, text="Reset", command=self.reset).pack(side='left', padx=5)
        ttk.Button(frame, text="Apply", command=self.apply).pack(side='left', padx=5)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.selected_idx = int(round(event.xdata))
        if 0 <= self.selected_idx < len(self.curve):
            self.update_point(event.ydata)

    def on_drag(self, event):
        if self.selected_idx is not None and event.inaxes == self.ax:
            self.update_point(event.ydata)

    def update_point(self, y):
        y = np.clip(y, 0, np.max(self.curve)*2)
        self.curve[self.selected_idx] = y
        self.line.set_ydata(self.curve)
        self.fig_canvas.draw()

    def normalize(self):
        max_val = np.max(np.abs(self.curve))
        if max_val > 0:
            self.curve = self.curve / max_val
            self.line.set_ydata(self.curve)
            self.fig_canvas.draw()

    def smooth(self):
        kernel = np.array([0.25, 0.5, 0.25])
        self.curve = np.convolve(self.curve, kernel, mode='same')
        self.line.set_ydata(self.curve)
        self.fig_canvas.draw()

    def quantize(self, num_points=20):
        length = len(self.curve)
        if num_points >= length:
            return

        x_indices = np.linspace(0, length - 1, num_points).astype(int)
        key_values = self.curve[x_indices]
        new_x = np.arange(length)
        quantized_curve = np.interp(new_x, x_indices, key_values)

        self.curve = quantized_curve
        self.line.set_ydata(self.curve)
        self.fig_canvas.draw()

    def reset(self):
        self.curve = self.original_curve.copy()
        self.line.set_ydata(self.curve)
        self.fig_canvas.draw()

    def apply(self):
        self.apply_callback(self.key, self.curve)
        self.master.destroy()
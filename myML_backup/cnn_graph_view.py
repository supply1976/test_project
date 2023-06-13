import os, sys
import tensorflow as tf
from tensorflow import keras
import tkinter as tk


class MyDraw(tk.Frame):
    def __init__(self, master, height=600, width=1000):
        super(MyDraw, self).__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, bg='#ffffff', 
            width=self.width, height=self.height)
        self.canvas.pack()
        self.pack()


def draw_conv2d_block(canvas_obj, 
        x0, y0, rec_w, rec_h, 
        text='Conv2D', ch_w=5, draw_skip=False):
    canvas_obj.create_rectangle(x0-rec_w//2, y0, x0+rec_w//2, y0+rec_h)
    canvas_obj.create_text(x0, y0+rec_h//2, text=text)
    canvas_obj.create_line(x0, y0+rec_h, x0, y0+rec_h+20, width=ch_w)
    yf = y0+rec_h+20
    if draw_skip:
        canvas_obj.create_arc(
            x0-rec_w, y0-10, x0+rec_w, y0+rec_h+10, 
            extent=180, start=90, style=tk.ARC)
    return (x0, yf)


if __name__=="__main__":
    root = tk.Tk()
    root.title("Test")
    mydraw = MyDraw(root)
    
    x0, y0 = 100, 50
    mydraw.canvas.create_oval(x0-3, y0-3, x0+3, y0+3, fill='black')
    mydraw.canvas.create_text(x0+10, y0-10, text='vt0 (input field)', anchor='nw')
    mydraw.canvas.create_line(x0, y0+3, x0, y0+23, width=1)
    coords_y = [y0, y0+23]
    for i, s in enumerate([60, 50, 60, 50, 70]):
        draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], s, s)
        yf = coords_y[-1] + s + 20
        coords_y.append(yf)
    draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], 20, 20, 
        text='1x1', ch_w=1, draw_skip=False)
    yf = coords_y[-1]
    yf = yf + 40
    mydraw.canvas.create_oval(x0-3, yf-3, x0+3, yf+3, fill='black')
    mydraw.canvas.create_text(x0+10, yf+3, text='cnn output', anchor='nw')
    #mydraw.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
    #        extent=180, start=90, style=tk.ARC)

    x0, y0 = 300, 150
    mydraw.canvas.create_text(x0, y0-30, text='resNetBlock', anchor='nw')
    mydraw.canvas.create_oval(x0-5, y0-5, x0+5, y0+5, fill='black')
    mydraw.canvas.create_text(x0+10, y0-10, text='x', anchor='nw')
    mydraw.canvas.create_line(x0, y0+5, x0, y0+25, width=5)
    coords_y = [y0, y0+23]
    for i, s in enumerate([50, 50, 40]):
        draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], s, s)
        yf = coords_y[-1] + s + 20
        coords_y.append(yf)
    #draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], 20, 20, 
    #    text='1x1', ch_w=1)
    yf = coords_y[-1]
    mydraw.canvas.create_oval(x0-5, yf-5, x0+5, yf+5, fill='black')
    mydraw.canvas.create_text(x0+10, yf+5, text='x+delta', anchor='nw') 
    mydraw.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
            extent=180, start=90, style=tk.ARC)
  
    x0, y0 = 450, 150
    #mydraw.canvas.create_text(x0, y0-30, text='resNetBlock', anchor='nw')
    mydraw.canvas.create_oval(x0-5, y0-5, x0+5, y0+5, fill='black')
    mydraw.canvas.create_text(x0+10, y0-10, text='x', anchor='nw')
    mydraw.canvas.create_line(x0, y0+5, x0, y0+25, width=5)
    coords_y = [y0, y0+23]
    for i, s in enumerate([50, 50]):
        draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], s, s)
        yf = coords_y[-1] + s + 20
        coords_y.append(yf)
    #draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], 20, 20, 
    #    text='1x1', ch_w=1)
    yf = coords_y[-1]
    mydraw.canvas.create_oval(x0-5, yf-5, x0+5, yf+5, fill='black')
    mydraw.canvas.create_text(x0+10, yf+5, text='x+delta', anchor='nw') 
    mydraw.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
            extent=180, start=90, style=tk.ARC)    

    x0, y0 = 600, 150
    #mydraw.canvas.create_text(x0, y0-30, text='resNetBlock', anchor='nw')
    mydraw.canvas.create_oval(x0-5, y0-5, x0+5, y0+5, fill='black')
    mydraw.canvas.create_text(x0+10, y0-10, text='x', anchor='nw')
    mydraw.canvas.create_line(x0, y0+5, x0, y0+25, width=5)
    coords_y = [y0, y0+23]
    for i, s in enumerate([50]):
        draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], s, s)
        yf = coords_y[-1] + s + 20
        coords_y.append(yf)
    #draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], 20, 20, 
    #    text='1x1', ch_w=1)
    yf = coords_y[-1]
    mydraw.canvas.create_oval(x0-5, yf-5, x0+5, yf+5, fill='black')
    mydraw.canvas.create_text(x0+10, yf+5, text='x+delta', anchor='nw') 
    mydraw.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
            extent=180, start=90, style=tk.ARC)    


    x0, y0 = 800, 100
    mydraw.canvas.create_oval(x0-3, y0-3, x0+3, y0+3, fill='black')
    mydraw.canvas.create_text(x0+10, y0-10, text='vt0 (input field)', anchor='nw')
    mydraw.canvas.create_line(x0, y0+3, x0, y0+23, width=1)
    coords_y = [y0, y0+23]
    for i, s in enumerate([70, 70, 70]):
        draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], s, s, 
            text="resNet_"+str(i))
        yf = coords_y[-1] + s + 20
        coords_y.append(yf)
    draw_conv2d_block(mydraw.canvas, x0, coords_y[-1], 20, 20, 
        text='1x1', ch_w=1, draw_skip=False)
    yf = coords_y[-1]
    yf = yf + 40
    mydraw.canvas.create_oval(x0-3, yf-3, x0+3, yf+3, fill='black')
    mydraw.canvas.create_text(x0+10, yf+3, text='cnn output', anchor='nw')
    mydraw.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
            extent=180, start=90, dash=(10,4), style=tk.ARC)

    mydraw.mainloop()


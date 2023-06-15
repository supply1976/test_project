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

    
    def draw_input_node(self, x0, y0, dr, input_text, 
        out_len, out_width, out_dir):
        """
        """
        self.canvas.create_oval(x0-dr, y0-dr, x0+dr, y0+dr, fill='black')
        self.canvas.create_text(x0, y0-10, text=input_text)
        x1, y1 = (0,0)
        if out_dir=='e':
            x1, y1 = x0+dr+out_len, y0
        elif out_dir=='s':
            x1, y1 = x0, y0+dr+out_len
        elif out_dir=='n':
            x1, y1 = x0, y0-dr-out_len
        else:
            x1, y1 = x0, y0
        self.canvas.create_line(x0, y0, x1, y1, width=out_width)
        return (x1, y1)

        
    def draw_conv2d_block(self, x0, y0, size=50, text='Conv2D', 
        out_len=20, out_width=4, out_dir='s'):
        """
        ul: upper-left
        lr: lower-right
        """
        if out_dir=='e':
            ulx, uly = (x0, y0-size//2)
            lrx, lry = (x0+size, y0+size//2)
            x1, y1 = (x0+size, y0)
            x2, y2 = (x1+out_len, y1)
        elif out_dir=='s':
            ulx, uly = (x0-size//2, y0)
            lrx, lry = (x0+size//2, y0+size)
            x1, y1 = (x0, y0+size)
            x2, y2 = (x1, y1+out_len)
        elif out_dir=='n':
            ulx, uly = (x0-size//2, y0-size)
            lrx, lry = (x0+size//2, y0)
            x1, y1 = (x0, y0-size)
            x2, y2 = (x1, y1-out_len)
        else:
            ulx, uly = (x0, y0)
            lrx, lry = (x0, x0)
            x1, y1 = (x0, y0)
        xc, yc = (ulx+lrx)//2, (uly+lry)//2
        self.canvas.create_rectangle(ulx, uly, lrx, lry)
        self.canvas.create_text(xc, yc, text=text)
        self.canvas.create_line(x1, y1, x2, y2, width=out_width)
        return (x2, y2)

        
    def draw_net(self, x0, y0, 
        add_skip, add_input, input_text, 
        rec_sizes, rec_text, out_len, out_width, 
        flowdir='x'):
        # draw input node and line
        dr, llen = (0, 0)
        if add_input:
            dr, llen, lw = (3, 30, 1)
        if flowdir=='x':
            x0, y0 = (x0+dr, y0)
            x1, y1 = (x0+dr+llen, y0)
        else:
            x0, y0 = (x0, y0+dr)
            x1, y1 = (x0, y0+dr+llen)        
        if add_input:
            self.canvas.create_oval(x0-dr, y0-dr, x0+dr, y0+dr, fill='black')
            self.canvas.create_text(x0, y0-10, text=input_text)
            self.canvas.create_line(x0, y0, x1, y1, width=lw)
        # stack of rec        
        for i, s in enumerate(rec_sizes):
            xc = x1+s//2 if flowdir=='x' else x1
            yc = y1 if flowdir=='x' else y1+s//2
            print(xc, yc)
            self.draw_conv2d_block(xc, yc, 
                size=s, text=rec_text, 
                width=out_width, llen=out_len, flowdir=flowdir)
            x1 = xc+s//2+out_len if flowdir=='x' else x1
            y1 = y1 if flowdir=='x' else yc+s//2+out_len
        
        if add_skip:
            y0 = y0+15 if add_input else y0-10
            self.canvas.create_arc(x0-60, y0, x0+60, y1-10, 
                extent=180, start=90, style=tk.ARC)
        return (x1, y1)

    
    def _input_conv2d_output(self, x0, y0, 
        line_len, 
        line_w, 
        input_text, 
        kerns, 
        rec_text,
        ch_w,
        draw_skip,
        skip_to_final=False):
        #
        self.canvas.create_oval(x0-3, y0-3, x0+3, y0+3, fill='black')
        self.canvas.create_text(x0+10, y0-10, text=input_text, anchor='nw')
        self.canvas.create_line(x0, y0, x0, y0+3+line_len, width=line_w)
        coords_y = [y0, y0+3+line_len]
        for i, s in enumerate(kerns):
            draw_conv2d_block(
               self.canvas, x0, coords_y[-1], s, s, 
               text=rec_text, ch_w=ch_w, draw_skip=draw_skip)
            yf = coords_y[-1] + s + line_len
            coords_y.append(yf)
        draw_conv2d_block(
            self.canvas, x0, coords_y[-1], 20, 20, 
            text='1x1', ch_w=1, draw_skip=False)
        yf = coords_y[-1]
        yf = yf + 40
        self.canvas.create_oval(x0-3, yf-3, x0+3, yf+3, fill='black')
        self.canvas.create_text(x0+10, yf+3, text='cnn output', anchor='nw')
        if skip_to_final:
            self.canvas.create_arc(x0-70, y0+10, x0+70, yf-10, 
                extent=180, start=90, dash=(10,4), style=tk.ARC)    




if __name__=="__main__":
    root = tk.Tk()
    root.title("")
    mydraw = MyDraw(root)
    
    x0, y0 = 100, 50
    x1, y1 = mydraw.draw_input_node(x0, y0, dr=3, 
        input_text="",
        out_len=30,
        out_width=1,
        out_dir='s')
    for _ in range(6):
        x1, y1 = mydraw.draw_conv2d_block(x1, y1)
   

    """
    flowdir = 'y'

    x0, y0 = 150, 50
    x1, y1 = mydraw.draw_net(x0, y0, 
             add_skip=True, add_input=True, input_text="vt0 (input image)",
             rec_sizes=[50,50, 50], rec_text="Conv2D", 
             out_len=20, out_width=4, flowdir=flowdir)
        
    x2, y2 = mydraw.draw_net(x1, y1, 
             add_skip=True, add_input=False, input_text="",
             rec_sizes=[50], rec_text="Conv2D", 
             out_len=20, out_width=4, flowdir=flowdir)
                 
    x3, y3 = mydraw.draw_net(x2, y2,
             add_skip=True, add_input=False, input_text="", 
             rec_sizes=[50, 50], rec_text='Conv2D',
             out_len=20, out_width=4, flowdir=flowdir)
    
    x4, y4 = mydraw.draw_net(x3-10, y3+10,
             add_skip=False, add_input=False, input_text="", 
             rec_sizes=[20], rec_text='1x1',
             out_len=100, out_width=1, flowdir='x')
             
    # add output node
    mydraw.canvas.create_oval(x4-3, y4-3, x4+3, y4+3, fill='black')
    mydraw.canvas.create_text(x4, y4+10, text="cnn output image")
    #mydraw.canvas.create_arc(x0-85, y0+10, x0+85, y4-10, 
    #    extent=180, start=90, dash=(10,4), style=tk.ARC)
        

    
    #x0, y0 = 100, 50
    #mydraw._input_conv2d_output(x0, y0, 
    #    line_len=20,
    #    line_w=1,
    #    input_text="vt0 (input images)",
    #    kerns=[40, 50, 50, 60, 70],
    #    rec_text="Conv2D",
    #    ch_w=5,
    #    draw_skip=True,
    #    skip_to_final=True)
    
    
    
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
    """
    mydraw.mainloop()


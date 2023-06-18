import os, sys
#import tensorflow as tf
#from tensorflow import keras
import tkinter as tk


class MyDraw(tk.Frame):
    def __init__(self, master, height=600, width=800):
        super(MyDraw, self).__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, bg='#ffffff', 
            width=self.width, height=self.height)
        self.canvas.pack()
        self.pack()

    
    def draw_node(self, x0, y0, dr, _text, 
        out_len, out_width, out_dir):
        """
        """
        self.canvas.create_oval(x0-dr, y0-dr, x0+dr, y0+dr, fill='black')
        #self.canvas.create_text(x0, y0-10, text=input_text)
        x1, y1 = (0,0)
        if out_dir=='e':
            x1, y1 = x0+dr+out_len, y0
            self.canvas.create_text(x0-15, y0-15, text=_text)
        elif out_dir=='s':
            x1, y1 = x0, y0+dr+out_len
            self.canvas.create_text(x0, y0-15, text=_text)
        elif out_dir=='n':
            x1, y1 = x0, y0-dr-out_len
            self.canvas.create_text(x0, y0+15, text=_text)
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
    
    def draw_skip_arc(self, x0, y0, x1, y1, dr, ifdash=False, out_dir='s'):
        if out_dir=='e':
            ulx, uly = x0, y0-dr
            lrx, lry = x1, y1+dr
            st, ext = 180, 180
        elif out_dir=='s':
            ulx, uly = x0-dr, y0
            lrx, lry = x1+dr, y1
            st, ext = 90, 180
        elif out_dir=='n':
            ulx, uly = x1-dr, y1
            lrx, lry = x0+dr, y0
            st, ext = 270, 180
        else:
            pass
        if ifdash:
            self.canvas.create_arc(ulx, uly, lrx, lry, start=st, dash=(4,4), extent=ext, style=tk.ARC)
        else:
            self.canvas.create_arc(ulx, uly, lrx, lry, start=st, extent=ext, style=tk.ARC)
            




if __name__=="__main__":
    root = tk.Tk()
    root.title("")
    mydraw = MyDraw(root, height=900, width=1200)
    flowdir = 's'
    
    # general CNN: sequential resNet
    x0, y0 = 100, 50
    x1, y1 = mydraw.draw_node(x0, y0, dr=3, _text="vt0 (input image)",
        out_len=30, out_width=1, out_dir=flowdir)
    x2, y2 = mydraw.draw_conv2d_block(x1, y1, out_dir=flowdir)
    x3, y3 = mydraw.draw_conv2d_block(x2, y2, out_dir=flowdir)
    x4, y4 = mydraw.draw_conv2d_block(x3, y3, out_dir=flowdir)
    x5, y5 = mydraw.draw_conv2d_block(x4, y4, out_dir=flowdir)
    x6, y6 = mydraw.draw_conv2d_block(x5, y5, out_dir=flowdir)
    x7, y7 = mydraw.draw_conv2d_block(x6, y6, out_dir=flowdir)
    mydraw.draw_skip_arc(x1, y1-10, x3, y3-10, dr=60, out_dir=flowdir)
    mydraw.draw_skip_arc(x3, y3-10, x5, y5-10, dr=60, out_dir=flowdir)
    mydraw.draw_skip_arc(x5, y5-10, x7, y7-10, dr=60, out_dir=flowdir)
    x8, y8 = mydraw.draw_conv2d_block(x7, y7, out_dir=flowdir, size=20, text='1x1', out_len=30, out_width=1)
    mydraw.draw_node(x8, y8, dr=3, _text="cnn output", 
        out_len=0, out_width=1, out_dir='n')
    #mydraw.draw_skip_arc(x0, y0+15, x8, y8-15, dr=70, ifdash=True, out_dir='n')

    # resNetBlock
    mydraw.canvas.create_text(250, 160, text="stack of resNetBlock")
    x0, y0 = 250, 200
    x1, y1 = mydraw.draw_node(x0, y0, dr=5, _text="x",
        out_len=30, out_width=3, out_dir=flowdir)
    x2, y2 = mydraw.draw_conv2d_block(x1, y1, text="resnet_1", out_dir=flowdir)
    x3, y3 = mydraw.draw_conv2d_block(x2, y2, text="resnet_2", out_dir=flowdir)
    x4, y4 = mydraw.draw_conv2d_block(x3, y3, text="resnet_3", out_len=30, out_width=3, out_dir=flowdir)  
    mydraw.draw_node(x4, y4, dr=5, _text="x+delta", 
        out_len=0, out_width=1, out_dir='n')

    # UNet
    x0, y0 = 500, 50
    x1, y1 = mydraw.draw_node(x0, y0, dr=3, _text="vt0 (input image)",
        out_len=30, out_width=1, out_dir=flowdir)
    x2, y2 = mydraw.draw_conv2d_block(x1, y1, out_dir=flowdir)
    x3, y3 = mydraw.draw_conv2d_block(x2, y2, out_dir=flowdir)
    x4, y4 = mydraw.draw_conv2d_block(x3, y3, out_dir=flowdir)
    x5, y5 = mydraw.draw_conv2d_block(x4, y4, out_dir=flowdir)
    x6, y6 = mydraw.draw_conv2d_block(x5, y5, out_dir=flowdir)
    x7, y7 = mydraw.draw_conv2d_block(x6-25, y6+25, out_len=60, out_dir='e')
    mydraw.draw_skip_arc(x1, y1-10, x3, y3-10, dr=60, out_dir=flowdir)
    mydraw.draw_skip_arc(x3, y3-10, x5, y5-10, dr=60, out_dir=flowdir)
    #mydraw.draw_skip_arc(x5-40, y5-10, x7-40, y7-10, dr=60, out_dir=flowdir)
    mydraw.canvas.create_arc(x5-60, y5-10, x7-20, y7+60, start=90, extent=253, style=tk.ARC)
    x8, y8 = mydraw.draw_conv2d_block(x7+10, y7+10, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    xf, yf = mydraw.draw_conv2d_block(x8, y8, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    xf, yf = mydraw.draw_conv2d_block(xf, yf, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    xf, yf = mydraw.draw_conv2d_block(xf, yf, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    xf, yf = mydraw.draw_conv2d_block(xf, yf, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    xf, yf = mydraw.draw_conv2d_block(xf, yf, out_dir='n', size=20, text='1x1', out_len=50, out_width=1)
    mydraw.canvas.create_line(x2, y2-10, x2+98, y2-10, arrow=tk.LAST)
    mydraw.canvas.create_line(x3, y3-10, x3+98, y3-10, arrow=tk.LAST)
    mydraw.canvas.create_line(x4, y4-10, x4+98, y4-10, arrow=tk.LAST)
    mydraw.canvas.create_line(x5, y5-10, x5+98, y5-10, arrow=tk.LAST)
    mydraw.canvas.create_line(x6, y6-10, x6+98, y6-10, arrow=tk.LAST)
    mydraw.draw_node(xf, yf, out_dir='s', dr=3, _text="cnn output", out_len=1, out_width=1)

    #mydraw.draw_node(x8, y8, dr=3, _text="cnn output", 
    #    out_len=0, out_width=1, out_dir='n')
    #mydraw.draw_skip_arc(x0, y0+15, x8, y8-15, dr=70, ifdash=True, out_dir='n')   
    
    
    
    
    mydraw.mainloop()


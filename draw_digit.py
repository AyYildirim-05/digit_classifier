import tkinter as tk
from PIL import Image, ImageDraw

canvas_size = 280
brush_size = 12

# creating a new canvas to draw on
image = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(image)

# function to handle drawing on the canvas
def paint(event):
    x, y = event.x, event.y
    draw.ellipse(
        (x - brush_size, y - brush_size,
         x + brush_size, y + brush_size),
        fill=255
    )
    canvas.create_oval(
        x - brush_size, y - brush_size,
        x + brush_size, y + brush_size,
        fill="white", outline="white"
    )

# saving the drawn image
def save():
    image.save("digit.png")
    print("Saved digit.png")

root = tk.Tk()
root.title("Draw a Digit")

# setting a black background for the canvas
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

canvas.bind("<B1-Motion>", paint)

button = tk.Button(root, text="Save", command=save)
button.pack()

root.mainloop()

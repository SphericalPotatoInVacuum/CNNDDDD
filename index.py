import tkinter as tk
import cnn
from PIL import Image, ImageDraw
import numpy as np
import math

prev_ev = None
model = None
image = None


def start_paint(event):
    global prev_ev
    x, y = event.x, event.y
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#000000")
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#000000")
    prev_ev = event
    predict(event)


def paint(event):
    global prev_ev
    x1, y1 = prev_ev.x, prev_ev.y
    x2, y2 = event.x, event.y
    canvas.create_line(x1, y1, x2, y2, fill="#000000", width=10)
    canvas.create_oval(x2 - 5, y2 - 5, x2 + 5, y2 + 5, fill="#000000")
    draw.line([(x1, y1), (x2, y2)], fill="#000000", width=10, joint="curve")
    draw.ellipse([(x2 - 5, y2 - 5), (x2 + 5, y2 + 5)], fill="#000000")
    prev_ev = event
    predict(event)


def start_erase(event):
    global prev_ev
    x, y = event.x, event.y
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                       fill="#FFFFFF", outline="#FFFFFF")
    draw.ellipse([x - 5, y - 5, x + 5, y + 5],
                 fill="#FFFFFF", outline="#FFFFFF")
    prev_ev = event
    predict(event)


def erase(event):
    global prev_ev
    x1, y1 = prev_ev.x, prev_ev.y
    x2, y2 = event.x, event.y
    canvas.create_line(x1, y1, x2, y2, fill="#FFFFFF", width=10)
    canvas.create_oval(x2 - 5, y2 - 5, x2 + 5, y2 + 5,
                       fill="#FFFFFF", outline="#FFFFFF")
    draw.line([(x1, y1), (x2, y2)], fill="#FFFFFF", width=10)
    draw.ellipse([(x2 - 5, y2 - 5), (x2 + 5, y2 + 5)],
                 fill="#FFFFFF", outline="#FFFFFF")
    prev_ev = event
    predict(event)


def predict(event):
    global image, image1, image2
    image2 = image1.resize((28, 28), resample=Image.BILINEAR)
    image = np.array(image2)
    image = np.vectorize(lambda x: (255 - x) / 255, otypes=[np.float])(image)
    image = np.apply_along_axis(
        lambda x: [x[0]], 2, image).reshape((1, 28, 28, 1))
    if model is None:
        label_message["text"] = "No models loaded or trained"
        return
    pred = model.predict(image)[0]
    for i in range(10):
        labels_prob[i]["text"] = "{}: {:05.2f}%".format(
            i, round(pred[i] * 100, 2))
        if pred[i] <= 0.5:
            labels_prob[i]["bg"] = "#ff{0:0{1}x}00".format(
                int(511 * pred[i]), 2)
        else:
            labels_prob[i]["bg"] = "#{0:0{1}x}ff00".format(
                int(511 * (1 - pred[i])), 2)
        labels_prob[i]["width"] = 7 + int(20 * pred[i])
    index = np.argmax(pred)
    label_message["text"] = ("Predicted number: {}, probability: {:05.2f}%".format(
        index, round(pred[index] * 100, 2)))


def load(event):
    global model
    model = cnn.create_model()
    try:
        model.load_weights("models/cnn")
        label_message["text"] = "Model loaded successfully"
    except Exception:
        label_message["text"] = "Model loading failed, no saved models found"


def train(event):
    global model
    label_message["text"] = "Training in progress..."
    label_message.update()
    cnn.load_data()
    (x_train, y_train), (x_test, y_test) = cnn.get_data()
    model = cnn.create_model()
    model.fit(x_train, y_train, batch_size=100,
              validation_data=(x_test, y_test))
    label_message["text"] = "Done! Now testing..."
    label_message.update()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    label_message["text"] = "Test accuracy: {}%".format(
        round(test_acc * 100, 2))


def save(event):
    model.save_weights("models/cnn")
    label_message["text"] = "Model saved"


def clear(event):
    global draw
    draw.rectangle([(1, 1), (280, 280)], fill="#FFFFFF")
    canvas.create_rectangle(0, 0, 282, 282, fill="#FFFFFF")
    image = None


root = tk.Tk()
root.resizable(width=False, height=False)
root.geometry('{}x{}'.format(600, 330))

frame_canvas = tk.Frame(root)
frame_canvas.grid(row=0, column=0)

label_canvas = tk.Label(frame_canvas, text="Press left mouse button to draw")
label_canvas.grid(row=1, column=0)

canvas = tk.Canvas(frame_canvas, width=280, height=280)
canvas.create_rectangle(1, 1, 282, 282, fill="#FFFFFF")
canvas.grid(row=0, column=0)
canvas.bind("<Button-1>", start_paint)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-3>", start_erase)
canvas.bind("<B3-Motion>", erase)

image1 = Image.new("RGB", (280, 280), (255, 255, 255))
draw = ImageDraw.Draw(image1)

frame_buttons = tk.Frame(root)
frame_buttons.grid(row=0, column=1)

button_load = tk.Button(frame_buttons, text="Load existing model")
button_load.grid(sticky='EW')
button_load.bind("<Button-1>", load)

button_train = tk.Button(frame_buttons, text="Train new model")
button_train.grid(sticky='EW')
button_train.bind("<Button-1>", train)

button_save = tk.Button(frame_buttons, text="Save model")
button_save.grid(sticky='EW')
button_save.bind("<Button-1>", save)

button_clear = tk.Button(frame_buttons, text="Clear")
button_clear.grid(sticky='EW')
button_clear.bind("<Button-1>", clear)

label_message = tk.Label(root, text="No messages")
label_message.grid(column=0, columnspan=4)

frame_probs = tk.Frame(root)
frame_probs.grid(row=0, column=2)

labels_prob = []
for i in range(10):
    labels_prob.append(tk.Label(
        frame_probs, text="{}: 0.00%".format(i), anchor=tk.W, justify=tk.LEFT))
    labels_prob[-1].grid(row=i, sticky="W")

root.mainloop()
exit(0)

cnn.load_data()
(x_train, y_train), (x_test, y_test) = cnn.get_data()

model = cnn.create_model()
model.summary()
model.fit(x_train, y_train, batch_size=100, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save_weights("models/cnn")

model = cnn.create_model()
model.load_weights("models/cnn")
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

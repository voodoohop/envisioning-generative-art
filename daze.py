print("hello")
from tqdm.notebook import trange
# from IPython.display import Image, display

from deep_daze import Imagine

TEXT = 'A slick high-contrast ad for an interactive futuristic school with children, teachers with a dark blue background' #@param {type:"string"}
NUM_LAYERS = 32 #@param {type:"number"}
SAVE_EVERY =  10#@param {type:"number"}
IMAGE_WIDTH =  256#@param {type:"number"}
SAVE_PROGRESS = True #@param {type:"boolean"}
LEARNING_RATE = 1e-5 #@param {type:"number"}
ITERATIONS = 1050 #@param {type:"number"}

model = Imagine(
    text = TEXT,
    num_layers = NUM_LAYERS,
    save_every = SAVE_EVERY,
    image_width = IMAGE_WIDTH,
    lr = LEARNING_RATE,
    iterations = ITERATIONS,
    save_progress = SAVE_PROGRESS
)

for epoch in trange(20, desc = 'epochs'):
    for i in trange(ITERATIONS, desc = 'iteration'):
        model.train_step(epoch, i)

        if i % model.save_every != 0:
            continue

        filename = TEXT[:77].replace(' ', '_')
        print("filename",filename)
        image = Image(f'./{filename}.jpg')
        display(image)
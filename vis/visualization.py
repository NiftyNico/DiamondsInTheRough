from q_functions import parse_arch
import numpy as np
import torch
import minerl
import matplotlib.pyplot as plt
from PIL import Image
import os


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def save_images(max_images=20, save_path='training_images'):
    """
    Generate max_images from user generated experiences to save_path
    :param max_images: number of 4-frame images to generate
    :param save_path: directory to save images
    :return: None
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    env_name = 'MineRLTreechopVectorObf-v0'
    data = minerl.data.make(env_name)
    for current_state, action, reward, next_state, done in data.batch_iter(batch_size=max_images, num_epochs=1, seq_len=4):
        # print(current_state['pov'].shape)   # max_images, seq_len, 64, 64, 3
        for img_num in range(max_images):
            # combine 4 frames into single image
            combined_frames = current_state['pov'][img_num, 0]
            for frame in range(1, 4):
                combined_frames = np.concatenate((combined_frames, current_state['pov'][img_num, frame]), axis=1)
            Image.fromarray(combined_frames).save(os.path.join(save_path, 'img_'+str(img_num)+'.png'))
        break


def get_convs(image, model_path, conv_layers=(0, 1, 2), save_path='conv_images'):
    """
    Saves images of conv layers in conv_layers to save_path as image passes through model in model_path
    :param image: input image   (numpy array)
    :param model_path: model path, e.g. "best/model.pt"
    :param conv_layers: indices of conv_layers to show
    :param save_path: directory to store images
    :return: None
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # stack image by channel after extracting the 4 frames in the image
    X = image[:, :64, :]
    for i in range(1, 4):
        X = np.concatenate((X, image[:, i*64:(i+1)*64, :]), axis=-1)
    X = np.swapaxes(X, 0, -1)   # channels first
    X = torch.Tensor(X)
    X = torch.unsqueeze(X, 0)  # add batch dimension

    # load model
    arch = 'dueling'
    n_actions = 30
    n_input_channels = 12
    model = parse_arch(arch, n_actions, n_input_channels)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # save each conv layer
    for layer in conv_layers:
        model.conv_layers[layer].register_forward_hook(get_activation('conv'+str(layer)))
        model(X)
        all_maps = activation['conv'+str(layer)][0, :, :, :].numpy()

        n = all_maps.shape[0]  # number of conv channels
        r, c = get_rc(n)  # rows, channels in subplot
        fig = plt.figure()
        # plt.set_cmap('Greys')
        # save each channel of current conv layer
        for i in range(n):
            curr_map = normalize(all_maps[i])
            img = Image.fromarray(curr_map)
            ax = fig.add_subplot(r, c, i + 1)
            ax.set_axis_off()
            ax.imshow(img)
        plt.savefig(os.path.join(save_path, 'conv_'+str(layer)+'.png'))


def normalize(img):
    # normalize conv values for display
    img = img - np.min(img)
    img = (img * 255 / np.max(img)).astype('uint8')
    return img


def get_rc(num):
    # determine subplot rows, columns
    row = np.floor(np.sqrt(num))
    while num % row != 0:
        row -= 1
    return int(row), int(num / row)


def main():
    # save_images() # only need to do this until you have an input image you like
    image = plt.imread(os.path.join('training_images', 'img_19.png'))   # image to run through network
    model_path = 'result/20201202T001742.297074/best/model.pt'
    get_convs(image, model_path)


if __name__ == '__main__':
    main()

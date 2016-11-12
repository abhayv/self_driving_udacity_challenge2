import os.path

import cv2
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.models import Sequential
from scipy.misc import imread

NUM_SPLITS = 1
TEST_SPLITS = [0]
TRAINING_SPLITS = []
SKIP_FRAMES = 20
NEED_TRANSPOSE = False

np.random.seed(1234)

# Common global variables
originalWidth = 640
originalHeight = 480
maxWidth = originalWidth
cutOffY = 200
maxHeight = originalHeight - cutOffY

img_width, img_height = 320, 120
num_classes = 102
max_output_value = 0.25
output_bins = np.arange(num_classes - 1) / (
(num_classes - 2) / 2.0 / max_output_value) - max_output_value
bin_width = 2.0 * max_output_value / (num_classes - 2)
num_channels = 1

all_datasets = ('/Volumes/Backup/datasets/2016-10-10-out/',
                '/Volumes/Backup/datasets/original-out/',
                '/Volumes/Backup/datasets/udacity-datasetElCaminoBack-out/',
                '/Volumes/Backup/datasets/dataset-2-2-b-out/',
                '/Volumes/Backup/datasets/udacity-datasetElCamino-out/')

challenge_training = ['/Volumes/Backup/datasets/challenge2/Train/%d/' % i for i in range(1, 7)]
challenge_training2 = ['/Volumes/Backup/datasets/challenge2/Train/%d/' % i for i in range(8, 20)]
challenge_training2b = ['/Volumes/Backup/datasets/challenge2/Train/%d/' % i for i in range(15, 20)]
challenge_test = ['/Volumes/Backup/datasets/challenge2/Test/']
original = ['/Volumes/Backup/datasets/original-out/']


def get_samples(root):
    """Gets samples from a root directory. The expected file structure is:
    root:
       interpolated.csv 
       center/1234.jpg
       center/1235.jpg
       ...
    where the contents of interpolated.csv are as follows:
    
    index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt
    2016-10-25,1477429515910314280,640,480,center_camera,center/1234.jpg,-0.0314159281552,0.184196478801,4.50321225868,0.0,0.0,0.0
    2016-10-25,1477429515910314280,640,480,center_camera,center/1235.jpg,-0.0314159281552,0.184196478801,4.50321225868,0.0,0.0,0.0
    ...
    
    Returns a pandas dataframe
    """
    df = pd.read_csv(root + '/interpolated.csv')
    
    # cut out non center camera and frames where car is not moving
    df = df[(df['frame_id'] == 'center_camera') & (df['speed'] > 0.1)]
    
    # smoothen the angles
    angles = np.mean(
        np.vstack((pd.ewma(df['angle'], com=20), pd.ewma(df['angle'][::-1], com=20)[::-1])), axis=0)
    df['angle'] = angles
    return df


def get_all(root):
    """Gets all samples from a root directory. Format is similar to get_samples."""
    df = pd.read_csv(root + '/interpolated.csv')
    return df


def bird_eye_view_homography():
    """
    Finds the homography for computing the bird's eye view.
    """
    topY = 100
    bottomX = 200

    rect = np.array([
        [0, topY],
        [maxWidth, topY],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - bottomX, maxHeight - 1],
        [bottomX, maxHeight - 1]], dtype="float32")

    h, mask = cv2.findHomography(rect, dst)
    return h


def bird_eye_view(img, homography):
    """Generate the bird's eye view using the homography.
     Currently not being used.
    """
    img = img[cutOffY:originalHeight, ...]
    warped = cv2.warpPerspective(img, homography, (maxWidth, maxHeight))
    warped = cv2.resize(warped, (img_width, img_height))
    return warped


def mask(image):
    """Mask out non interesting parts of the image.
    We try to keep the lanes in the region of interest.
    """
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask1 = np.zeros_like(image)
    mask2 = np.zeros_like(image)
    channel_count = 1
    if len(image.shape) > 2:
        channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices1 = np.array([[(0, imshape[0]), (0, imshape[0] * 3.1 / 4),
                           (imshape[1] * 3.5 / 9, imshape[0] * 1 / 2),
                           (imshape[1] * 5.5 / 9, imshape[0] * 1 / 2),
                           (imshape[1] * 0.1, imshape[0]), (0, imshape[0])]], dtype=np.int32)
    vertices2 = np.array([[(imshape[1] * 8.5 / 9.0, imshape[0]),
                           (imshape[1] * 4.5 / 9, imshape[0] * 1 / 2),
                           (imshape[1] * 5.5 / 9, imshape[0] * 1 / 2),
                           (imshape[1], imshape[0] * 3.0 / 4), (imshape[1], imshape[0])]],
                         dtype=np.int32)
    cv2.fillPoly(mask1, vertices1, ignore_mask_color)
    cv2.fillPoly(mask2, vertices2, ignore_mask_color)

    image1 = cv2.bitwise_and(image, mask1)
    image2 = cv2.bitwise_and(image, mask2)
    image = cv2.bitwise_or(image1, image2)

    if channel_count > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray[imshape[0] * 1 / 2:imshape[0], ...]
    gray = cv2.resize(gray, (img_width, img_height))
    return gray


def hough2(img):
    """Find lines using Hough transform.
    Currently not being used.
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 20, 150, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 0.1, np.pi / 1800, 1, 1, 2)
    try:
        range = lines.shape[0]
    except AttributeError:
        range = 0

    for i in xrange(range):
        for x1, y1, x2, y2 in lines[i]:
            if x2 != x1:
                if abs(y2 - y1) / abs(x2 - x1) > 0.1:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    img = cv2.bitwise_or(edges, img)
    return img


def denormalize(img):
    img = (img * 255.0).astype('uint8')
    if NEED_TRANSPOSE:
        img = np.transpose(img, axes=(1, 2, 0))
    return img[0]


def pre_process(img, homography):
    img = mask(img)

    img = img / 255.0
    img = img.astype(np.float32)
    img = np.array([img])
    if NEED_TRANSPOSE:
        img = np.transpose(img, axes=(2, 0, 1))
    return img


def process_output(angle_class):
    one_hot = np.zeros(num_classes, dtype=int)
    one_hot[np.digitize(angle_class, output_bins)] = 1
    return one_hot


def unprocess_output(angle_class):
    hot_bin = np.argmax(angle_class)
    if hot_bin < len(output_bins):
        return output_bins[hot_bin] - bin_width / 2.0
    else:
        return max_output_value + bin_width / 2.0


def generator(root, df_split, batch_size, homography, skip_frames=SKIP_FRAMES,
              return_originals=False):
    """
    Generator for Keras model training and testing.
    :param root: directory with data.
    :param df_split: the part of the data frame to sample from
    :param batch_size: for keras
    :param homography: pass along to pre_process
    :param skip_frames: number of frames to skip
    :param return_originals: boolean to indicate if the originals should be returned
    :return:
    """
    starting_pos = 0
    while True:
        image_names, originals, images, original_angles, angles = [], [], [], [], []
        split = np.array(df_split)
        rolled = np.roll(split, -starting_pos, axis=0)
        batch_samples = rolled[0:batch_size * skip_frames:skip_frames, ...]
        starting_pos = (batch_size * skip_frames + starting_pos) % split.shape[0]
        for sample in batch_samples:
            img_name, angle_class = sample[5], sample[6]
            original_angle = angle_class
            angle_class = process_output(angle_class)
            try:
                original = imread(root + img_name)
            except Exception as e:
                print "Exception reading file", e
                continue
            img = original
            if img is None:
                print 'img is none'
                continue
            img = pre_process(img, homography)
            if img.shape != (num_channels, img_height, img_width):
                print 'img shape is wrong', img.shape
            originals.append(original)
            image_names.append(img_name)
            images.append(img)
            angles.append(angle_class)
            original_angles.append(original_angle)
        batch_images = np.array(images)
        batch_angles = np.array(angles)
        if return_originals:
            yield image_names, originals, original_angles, batch_images, batch_angles
        else:
            yield batch_images, batch_angles


def get_model():
    """
    Gets the model for Keras.
    :return:
    """
    model = Sequential()

    image_input_shape = (num_channels, img_height, img_width)
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=image_input_shape))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(32, 3, 3))

    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    frame_model = model
    frame_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return frame_model


def predict(root, model, test_splits, viz_bad_ones=0, skip_frames=10, output_values=False,
            num_batches=3, batch_size=160, max_num=6601):
    """
    Predict using the learned model.
    :param root: root directory
    :param model: Keras model
    :param test_splits: data frame portion to use for testing
    :param viz_bad_ones:
    :param skip_frames:
    :param output_values:
    :param num_batches:
    :param batch_size:
    :param max_num: stop after processing these many frames.
    :return:
    """
    homography = bird_eye_view_homography()
    test_generator = generator(root, test_splits, batch_size, homography, skip_frames=skip_frames,
                               return_originals=True)
    if output_values:
        num_output = 0
        output_values_array = []
        output_names_array = []

    for batch in range(num_batches):
        image_names, originals, original_angles, test_x, test_y = test_generator.next()
        yhat = model.predict(test_x)
        yhat = np.array([unprocess_output(y) for y in yhat])
        if output_values:
            for i in range(len(originals)):
                image_name = image_names[i]
                image_name = image_name.split('/')[-1].split('.')[0]
                y = yhat[i]
                num_output += 1
                output_values_array.append(y)
                output_names_array.append(image_name)
                if num_output == max_num:
                    return pd.DataFrame(data={'frame_id': output_names_array,
                                              'steering_angle': output_values_array})
        else:
            rmse = np.sqrt(np.mean((yhat - original_angles) ** 2))
            print("model evaluated RMSE:", rmse)

            if viz_bad_ones:
                for y1, y2, x in zip(yhat, original_angles, test_x):
                    if abs(y1 - y2) > viz_bad_ones:
                        x = denormalize(x)
                        print y1, y2
                        cv2.imshow('p', x)
                        cv2.waitKey(0)

            plt.figure(figsize=(32, 8))
            plt.plot(original_angles, 'r.-', label='target')
            plt.plot(yhat, 'b^-', label='predict')
            plt.legend(loc='best')
            plt.title("RMSE: %.2f" % rmse)
            output = "predicted%s.png" % batch
            plt.savefig(output)


def do_predict(root):
    model = get_model()
    model.load_weights('model.h5')
    test_splits = get_samples(root)
    predict(root, model, test_splits, viz_bad_ones=0.00001)


def predict_all(test_root, max_num=6601):
    model = get_model()
    model.load_weights('model.h5')
    test_splits = get_samples(test_root)

    output_df = predict(test_root, model, test_splits, viz_bad_ones=0, skip_frames=1,
                        num_batches=100, batch_size=67,
                        output_values=True, max_num=max_num)
    output_df.to_csv('predictions.csv', index=False)


def train(root, model, training_splits):
    """
    Train the Keras model.
    :param root:
    :param model:
    :param training_splits:
    :return:
    """
    nb_train_samples = 500
    nb_validation_samples = 100
    nb_epoch = 10
    batch_size = 1

    homography = bird_eye_view_homography()
    train_generator = generator(root, training_splits, batch_size, homography)

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose=True,
        nb_val_samples=nb_validation_samples)


def do_train(root):
    model = get_model()
    if os.path.isfile('model.h5'):
        model.load_weights('model.h5')
    training_splits = get_samples(root)
    train(root, model, training_splits)
    save_model(model)
    model.save_weights('model.h5')


def do_visualize(root):
    splits = get_samples(root)
    homography = bird_eye_view_homography()
    for image_names, originals, original_angles, samples, angles in generator(root, splits, 100,
                                                                              homography,
                                                                              skip_frames=10,
                                                                              return_originals=True):
        print len(image_names), samples.shape, angles.shape
        for i in range(len(originals)):
            original_angle = original_angles[i]
            image_name = image_names[i]
            angle = angles[i]
            angle = unprocess_output(angle)
            print image_name, angle, original_angle
            original = originals[i]
            sample = samples[i]
            sample = denormalize(sample)
            cv2.imshow("original", original)
            cv2.imshow("sample", sample)
            cv2.waitKey(0)


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


def test_preprocess(root):
    homography = bird_eye_view_homography()
    for filename in os.listdir(root + '/center'):
        img = cv2.imread(root + '/center/' + filename)
        cv2.imshow('n', img)
        x1 = pre_process(img, homography)
        y = denormalize(x1)
        cv2.imshow('d', y)
        cv2.waitKey(0)


def test_bins():
    print output_bins
    print bin_width
    for angle in [-1.5, -1, -0.9, -0.5, -.2, 0, 0.2, 0.5, 1.0, 1.2]:
        processed = process_output(angle)
        print angle, processed, unprocess_output(processed)


@click.command()
@click.option('--predict/--no-predict', default=False, help='do prediction')
@click.option('--train/--no-train', default=False, help='do training')
@click.option('--visualize/--no-visualize', default=False, help='visualize')
def generate_cli(predict, train, visualize):
    test_root = '/Volumes/Backup/datasets/challenge2/Train/20/'
    for root in [test_root]:
        print 'Now doing', root
        if visualize:
            do_visualize(root)
        if predict:
            do_predict(root)
        if train:
            do_train(root)


if __name__ == '__main__':
    generate_cli()

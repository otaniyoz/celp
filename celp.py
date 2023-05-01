import sys, os
from ctypes import *
import math
import random

lib = CDLL(os.path.join(os.path.dirname(__file__), "libcelp.so"), RTLD_GLOBAL)

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class IMAGE(Structure):
    _fields_ = [("c", c_int),
                ("h", c_int),
                ("w", c_int),
                ("data", POINTER(c_float))]
    def __add__(self, other):
        return add_image(self, other)
    def __sub__(self, other):
        return sub_image(self, other)

class POINT(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class DESCRIPTOR(Structure):
    _fields_ = [("p", POINT),
                ("n", c_int),
                ("data", POINTER(c_float))]

class MATRIX(Structure):
    _fields_ = [("rows", c_int),
                ("cols", c_int),
                ("data", POINTER(POINTER(c_double))),
                ("shallow", c_int)]

class DATA(Structure):
    _fields_ = [("X", MATRIX),
                ("y", MATRIX)]

class LAYER(Structure):
    _fields_ = [("in", MATRIX),
                ("dw", MATRIX),
                ("w", MATRIX),
                ("v", MATRIX),
                ("out", MATRIX),
                ("activation", c_int)]

class MODEL(Structure):
    _fields_ = [("layers", POINTER(LAYER)),
                ("n", c_int)]


(LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX) = range(5)


add_image = lib.add_image
add_image.argtypes = [IMAGE, IMAGE]
add_image.restype = IMAGE

sub_image = lib.sub_image
sub_image.argtypes = [IMAGE, IMAGE]
sub_image.restype = IMAGE

mult_image = lib.mult_image
mult_image.argtypes = [IMAGE, IMAGE]
mult_image.restype = IMAGE

invert_image = lib.invert_image
invert_image.argtypes = [IMAGE]
invert_image.restype = None

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]

get_pixel = lib.get_pixel
get_pixel.argtypes = [IMAGE, c_int, c_int, c_int]
get_pixel.restype = c_float

set_pixel = lib.set_pixel
set_pixel.argtypes = [IMAGE, c_int, c_int, c_int, c_float]
set_pixel.restype = None

rgb_to_grayscale = lib.rgb_to_grayscale
rgb_to_grayscale.argtypes = [IMAGE]
rgb_to_grayscale.restype = IMAGE

grayscale_to_rgb = lib.grayscale_to_rgb
grayscale_to_rgb.argtypes = [IMAGE]
grayscale_to_rgb.restype = IMAGE

copy_image = lib.copy_image
copy_image.argtypes = [IMAGE]
copy_image.restype = IMAGE

rgb_to_hsv = lib.rgb_to_hsv
rgb_to_hsv.argtypes = [IMAGE]
rgb_to_hsv.restype = None

feature_normalize = lib.feature_normalize
feature_normalize.argtypes = [IMAGE]
feature_normalize.restype = None

clamp_image = lib.clamp_image
clamp_image.argtypes = [IMAGE]
clamp_image.restype = None

hsv_to_rgb = lib.hsv_to_rgb
hsv_to_rgb.argtypes = [IMAGE]
hsv_to_rgb.restype = None

shift_image = lib.shift_image
shift_image.argtypes = [IMAGE, c_int, c_float]
shift_image.restype = None

scale_image = lib.scale_image
scale_image.argtypes = [IMAGE, c_int, c_float]
scale_image.restype = None

load_image_lib = lib.load_image
load_image_lib.argtypes = [c_char_p]
load_image_lib.restype = IMAGE

def load_image(f):
    return load_image_lib(f.encode('ascii'))

save_png_lib = lib.save_png
save_png_lib.argtypes = [IMAGE, c_char_p]
save_png_lib.restype = None

def save_png(im, f):
    return save_png_lib(im, f.encode('ascii'))

save_image_lib = lib.save_image
save_image_lib.argtypes = [IMAGE, c_char_p]
save_image_lib.restype = None

def save_image(im, f):
    return save_image_lib(im, f.encode('ascii'))

same_image = lib.same_image
same_image.argtypes = [IMAGE, IMAGE]
same_image.restype = c_int

nn_resize = lib.nn_resize
nn_resize.argtypes = [IMAGE, c_int, c_int]
nn_resize.restype = IMAGE

bilinear_resize = lib.bilinear_resize
bilinear_resize.argtypes = [IMAGE, c_int, c_int]
bilinear_resize.restype = IMAGE

make_sharpen_filter = lib.make_sharpen_filter
make_sharpen_filter.argtypes = []
make_sharpen_filter.restype = IMAGE

make_box_filter = lib.make_box_filter
make_box_filter.argtypes = [c_int]
make_box_filter.restype = IMAGE

make_emboss_filter = lib.make_emboss_filter
make_emboss_filter.argtypes = []
make_emboss_filter.restype = IMAGE

make_highpass_filter = lib.make_highpass_filter
make_highpass_filter.argtypes = []
make_highpass_filter.restype = IMAGE

make_gy_filter = lib.make_gy_filter
make_gy_filter.argtypes = []
make_gy_filter.restype = IMAGE

make_gx_filter = lib.make_gx_filter
make_gx_filter.argtypes = []
make_gx_filter.restype = IMAGE

sobel_image = lib.sobel_image
sobel_image.argtypes = [IMAGE]
sobel_image.restype = POINTER(IMAGE)

colorize_sobel = lib.colorize_sobel
colorize_sobel.argtypes = [IMAGE]
colorize_sobel.restype = IMAGE

make_gaussian_filter = lib.make_gaussian_filter
make_gaussian_filter.argtypes = [c_float]
make_gaussian_filter.restype = IMAGE

smooth_image = lib.smooth_image
smooth_image.argtypes = [IMAGE, c_float]
smooth_image.restype = IMAGE

convolve_image = lib.convolve_image
convolve_image.argtypes = [IMAGE, IMAGE, c_int]
convolve_image.restype = IMAGE

harris_corner_detector = lib.harris_corner_detector
harris_corner_detector.argtypes = [IMAGE, c_float, c_float, c_int, POINTER(c_int)]
harris_corner_detector.restype = POINTER(DESCRIPTOR)

mark_corners = lib.mark_corners
mark_corners.argtypes = [IMAGE, POINTER(DESCRIPTOR), c_int]
mark_corners.restype = None

detect_and_draw_corners = lib.detect_and_draw_corners
detect_and_draw_corners.argtypes = [IMAGE, c_float, c_float, c_int]
detect_and_draw_corners.restype = None

cylindrical_project = lib.cylindrical_project
cylindrical_project.argtypes = [IMAGE, c_float]
cylindrical_project.restype = IMAGE

structure_matrix = lib.structure_matrix
structure_matrix.argtypes = [IMAGE, c_float]
structure_matrix.restype = IMAGE

find_and_draw_matches = lib.find_and_draw_matches
find_and_draw_matches.argtypes = [IMAGE, IMAGE, c_float, c_float, c_int]
find_and_draw_matches.restype = IMAGE

panorama_image_lib = lib.panorama_image
panorama_image_lib.argtypes = [IMAGE, IMAGE, c_float, c_float, c_int, c_float, c_int, c_int]
panorama_image_lib.restype = IMAGE

draw_flow = lib.draw_flow
draw_flow.argtypes = [IMAGE, IMAGE, c_float]
draw_flow.restype = None

box_filter_image = lib.box_filter_image
box_filter_image.argtypes = [IMAGE, c_int]
box_filter_image.restype = IMAGE

optical_flow_images = lib.optical_flow_images
optical_flow_images.argtypes = [IMAGE, IMAGE, c_int, c_int]
optical_flow_images.restype = IMAGE

optical_flow_webcam = lib.optical_flow_webcam
optical_flow_webcam.argtypes = [c_int, c_int, c_int]
optical_flow_webcam.restype = None

def panorama_image(a, b, sigma=2, thresh=5, nms=3, inlier_thresh=2, iters=10000, cutoff=30):
    return panorama_image_lib(a, b, sigma, thresh, nms, inlier_thresh, iters, cutoff)


train_model = lib.train_model
train_model.argtypes = [MODEL, DATA, c_int, c_int, c_double, c_double, c_double]
train_model.restype = None

accuracy_model = lib.accuracy_model
accuracy_model.argtypes = [MODEL, DATA]
accuracy_model.restype = c_double

forward_model = lib.forward_model
forward_model.argtypes = [MODEL, MATRIX]
forward_model.restype = MATRIX

load_classification_data = lib.load_classification_data
load_classification_data.argtypes = [c_char_p, c_char_p, c_int]
load_classification_data.restype = DATA

make_layer = lib.make_layer
make_layer.argtypes = [c_int, c_int, c_int]
make_layer.restype = LAYER

def make_model(layers):
    m = MODEL()
    m.n = len(layers)
    m.layers = (LAYER*m.n) (*layers)
    return m

get_canny_edges = lib.get_canny_edges
get_canny_edges.argtypes = [IMAGE, c_float, c_float, c_float]
get_canny_edges.restype = IMAGE

get_xdog_edges = lib.get_xdog_edges
get_xdog_edges.argtypes = [IMAGE, c_float, c_float, c_float, c_float, c_float]
get_xdog_edges.restype = IMAGE

get_xfdog_edges = lib.get_xfdog_edges
get_xfdog_edges.argtypes = [IMAGE, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
get_xfdog_edges.restype = IMAGE

def xfdog_on_video(filename):
  se = 0.6 
  sc = 0.8
  sm = 0.8
  sa = 0.8
  k = 1.5#12.5 
  p = 180
  phi = 180
  e = 0.95

  frame =0 
  vid = cv2.VideoCapture(filename)
  while True:
    r, f = vid.read(frame)
    f = f / 255 if numpy.max(f) == 255 else f
    if not r: break

    c, h, w = f.shape[2],f.shape[1],f.shape[0]
    im = numpy.zeros((h,w,c), dtype=float)
    im[...,0] = f[...,2]
    im[...,1] = f[...,0]
    im[...,2] = f[...,1]
    image = make_image(c, h, w)
    arr = []
    for ci in range(c):
      for hi in range(h):
        for wi in range(w):
          arr.append(im[hi,wi,ci])
    c_arr = c_array(c_float,arr) 
    for i in range(c*h*w): image.data[i] = c_arr[i] 

    image = bilinear_resize(image, h//2, w//2)
    edges = get_xfdog_edges(image, se, sc, sm, sa, k, p, e, phi)
    save_image(edges, f"frames/{frame:003d}")
    free_image(edges)
    free_image(image)
    frame += 1


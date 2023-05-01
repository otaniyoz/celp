#ifndef IMAGE_H
#define IMAGE_H
#include <stdio.h>

#include "matrix.h"
#define TWOPI 6.2831853
#define PI 3.1415925

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int c, h, w;
  float *data;
} image;

// A 2d point.
// float x, y: the coordinates of the point.
typedef struct {
  float x, y;
} point;

// A descriptor for a point in an image.
// point p: x,y coordinates of the image pixel.
// int n: the number of floating point values in the descriptor.
// float *data: the descriptor for the pixel.
typedef struct {
  point p;
  int n;
  float *data;
} descriptor;

// A match between two points in an image.
// point p, q: x,y coordinates of the two matching pixels.
// int ai, bi: indexes in the descriptor array. For eliminating duplicates.
// float distance: the distance between the descriptors for the points.
typedef struct {
  point p, q;
  int ai, bi;
  float distance;
} match;

// Basic operations
float get_pixel(image im, int c, int h, int w);
void set_pixel(image im, int c, int h, int w, float v);
image copy_image(image im);
image rgb_to_grayscale(image im);
image grayscale_to_rgb(image im, float r, float g, float b);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void rgb_to_xyz(image im);
void xyz_to_rgb(image im);
void xyz_to_lab(image im);
void lab_to_xyz(image im);
void shift_image(image im, int c, float v);
void scale_image(image im, int c, float v);
void clamp_image(image im);
void correct_gamma(image im, float gamma, float A);
image get_channel(image im, int c);
int same_image(image a, image b, float eps);
image sub_image(image a, image b);
image add_image(image a, image b);
image blend_image(image a, image b, float k);
int *get_histogram_image(image im, int n);

// Loading and saving
image float_to_image(float *data, int c, int h, int w);
image make_image(int c, int h, int w);
image load_image(char *filename);
void save_image(image im, const char *name);
void save_png(image im, const char *name);
void save_image_binary(image im, const char *fname);
image load_image_binary(const char *fname);
void save_png(image im, const char *name);
void free_image(image im);

// Resizing
float nn_interpolate(image im, int c, float h, float w);
image nn_resize(image im, int h, int w);
float bilinear_interpolate(image im, int c, float h, float w);
image bilinear_resize(image im, int h, int w);

// Filtering
image convolve_image(image im, image filter, int preserve);
image make_box_filter(int w);
image make_highpass_filter();
image make_sharpen_filter();
image make_emboss_filter();
image make_gaussian_filter(float sigma);
image make_1d_gaussian(float sigma);
image make_gx_filter();
image make_gy_filter();
void feature_normalize(image im);
void l1_normalize(image im);
void threshold_image(image im, float thresh);
image *sobel_image(image im);
image colorize_sobel(image im);
image smooth_image(image im, float sigma);
image convolve_line_integral(image vectors, image im, image filter);
void sort_pixels(image im, float low, float high, int axis, int mode);

// Harris and Stitching
point make_point(float x, float y);
point project_point(matrix H, point p);
matrix compute_homography(match *matches, int n);
image structure_matrix(image im, float sigma);
image cornerness_response(image S);
void free_descriptors(descriptor *d, int n);
image cylindrical_project(image im, float f);
void mark_corners(image im, descriptor *d, int n);
image find_and_draw_matches(image a, image b, float sigma, float thresh,
                            int nms);
void detect_and_draw_corners(image im, float sigma, float thresh, int nms);
int model_inliers(matrix H, match *m, int n, float thresh);
image combine_images(image a, image b, matrix H);
match *match_descriptors(descriptor *a, int an, descriptor *b, int bn, int *mn);
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms,
                                   int *n);
image panorama_image(image a, image b, float sigma, float thresh, int nms,
                     float inlier_thresh, int iters, int cutoff);

// Edges
image get_canny_edges(image im, float sigma, float low, float high);
image get_xdog_edges(image im, float se, float sc, float sm, float sa, float k,
                     float p, float e, float phi, int flow);

// Quantization
void apply_otsu_threshold(image im, int n);
void floyd_steinberg_dither(image im, image q);

#ifdef __cplusplus
}
#endif
#endif

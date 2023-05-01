#include "image.h"
#include <math.h>
#include <stdlib.h>

// Extracts edges from an input image using Canny edge detection
// https://en.wikipedia.org/wiki/Canny_edge_detector
// image im: input image
// float sigma: standard deviation of the gaussian blur
// float low: lower double threshold
// float high: upper double threshold
// returns: image edges containing Canny edges
// TODO: looks like i made a mistake
image get_canny_edges(image im, float sigma, float low, float high) {
  float mag, angle;
  int j, i, x1, x2, y1, y2;
  image edges = make_image(im.c, im.h, im.w);

  if (im.c == 3) {
    im = rgb_to_grayscale(im);
  }
  // Apply Gaussian blur to remove the noise
  im = smooth_image(im, sigma);

  // Find the intensity gradients using the Sobel operator
  image *sobel = sobel_image(im);
  feature_normalize(sobel[0]);

  // Apply gradient magnitude thresholding
  // using discretized angles
  for (i = 1; i <= im.w; i++) {
    x1 = 0;
    x2 = 0;
    for (j = 1; j <= im.h; j++) {
      y1 = 0;
      y2 = 0;
      mag = get_pixel(sobel[0], 0, j, i);
      angle = get_pixel(sobel[1], 0, j, i) * 57.2958;
      if ((angle <= 22.5 && angle > 0.0) || (angle <= 180 && angle > 157.5)) {
        x1 = i + 1;
        x2 = i - 1;
        y1 = j;
        y2 = j;
      } else if (angle <= 67.5 && angle > 22.5) {
        x1 = i + 1;
        y1 = j + 1;
        x2 = i - 1;
        y2 = j - 1;
      } else if (angle <= 90.0 && angle > 67.5) {
        x1 = i;
        x2 = i;
        y1 = j + 1;
        y2 = j - 1;
      } else if (angle <= 135.0 && angle > 90.0) {
        x1 = i + 1;
        y1 = j - 1;
        x2 = i - 1;
        y2 = j + 1;
      }

      // Suppress pixels with the smaller magnitude
      // among two neighbours in the same direction
      // Then apply double threshold to mark weak
      // and strong edge pixels
      // Finally track down weak edge pixels
      // to decide if they should be suppressed or not
      if (mag < get_pixel(sobel[0], 0, y1, x1) ||
          mag < get_pixel(sobel[0], 0, y2, x2) || mag < low) {
        set_pixel(edges, 0, j, i, 0.0);
      } else {
        if (mag >= high) {
          set_pixel(edges, 0, j, i, 1.0);
        } else {
          float n1 = get_pixel(sobel[0], 0, j + 1, i - 1);
          float n2 = get_pixel(sobel[0], 0, j + 1, i);
          float n3 = get_pixel(sobel[0], 0, j + 1, i + 1);
          float n4 = get_pixel(sobel[0], 0, j, i - 1);
          float n5 = get_pixel(sobel[0], 0, j, i + 1);
          float n6 = get_pixel(sobel[0], 0, j - 1, i - 1);
          float n7 = get_pixel(sobel[0], 0, j - 1, i);
          float n8 = get_pixel(sobel[0], 0, j - 1, i + 1);
          if (n1 >= high || n2 >= high || n3 >= high || n4 >= high ||
              n5 >= high || n6 >= high || n7 >= high || n8 >= high) {
            set_pixel(edges, 0, j, i, 1.0);
          } else {
            set_pixel(edges, 0, j, i, 0.0);
          }
        }
      }
    }
  }
  free_image(sobel[0]);
  free_image(sobel[1]);
  free(sobel);
  return edges;
}

// Applies extended difference of gaussians to an image
// https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf
// image im: input image
// float se: standard deviation of the blur in xdog
// float sc: standard deviation of the blur in structure tensor
// float sm: standard deviation of the blur in line integral convolution
// float sa: standard deviation of the blur in edge lines
// float k: scale of the difference between two gaussians
// float p: scale of the edge prominence
// float e: threshold of clamping to white
// float phi: curve of the hyperbolic tangent used in thresholding
// int flow: whether to use the flow or not
// returns: image S containing edges
image get_xdog_edges(image im, float se, float sc, float sm, float sa, float k,
                     float p, float e, float phi, int flow) {
  int j, i;
  float xx, yy, xy, l, s;

  image D1 = smooth_image(im, se);
  image D2 = smooth_image(im, k * se);
  image S = make_image(1, im.h, im.w);
  for (i = 0; i < im.h * im.w; i++) {
    s = (1.0 + p) * D1.data[i] - p * D2.data[i];
    S.data[i] = (s >= e) ? 1.0 : 1.0 + tanhf(phi * s - e);
  }

  if (flow) {
    image ST = structure_matrix(im, sc);
    image R = make_image(2, ST.h, ST.w);
    for (i = 0; i < ST.w; i++) {
      for (j = 0; j < ST.h; j++) {
        xx = get_pixel(ST, 0, j, i);
        yy = get_pixel(ST, 1, j, i);
        xy = get_pixel(ST, 2, j, i);
        l = xx + yy + 0.5 * sqrtf((xx - yy) * (xx - yy) + 4.0 * xy * xy);
        set_pixel(R, 0, j, i, (l - xx) / ((l - xx) * (l - xx) + xy * xy));
        set_pixel(R, 1, j, i, -xy / ((l - xx) * (l - xx) + xy * xy));
      }
    }

    image gm = make_1d_gaussian(sm);
    image ga = make_1d_gaussian(sa);
    S = convolve_line_integral(R, S, gm);
    S = convolve_line_integral(R, S, ga);
    S = convolve_line_integral(R, S, ga);
    for (i = 0; i < S.h * S.w; i++) {
      S.data[i] = (S.data[i] >= e) ? 1.0 : 1.0 + tanhf(phi * S.data[i] - e);
    }

    free_image(ST);
    free_image(gm);
    free_image(ga);
    free_image(R);
  }
  free_image(D1);
  free_image(D2);
  return S;
}
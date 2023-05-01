#include <math.h>
#include "image.h"

float nn_interpolate(image im, int c, float h, float w)
{
  return get_pixel(im, c, floor(h + 0.5), floor(w + 0.5));
}

image nn_resize(image im, int h, int w)
{
  float y, x;
  float xs = (float)im.w/w;
  float ys = (float)im.h/h;
  int k, j, i;
  image resized = make_image(im.c, h, w);
  for (k = 0; k < im.c; k++) {
    for (j = 0; j < h; j++) {
      y = (j+0.5)*ys - 0.5; 
      for (i = 0; i < w; i++) {
        x = (i+0.5)*xs - 0.5;
        set_pixel(resized, k, j, i, nn_interpolate(im, k, y, x));
      }
    }
  }
  return resized;
}

float bilinear_interpolate(image im, int c, float h, float w)
{
  float top, bottom, left, right, d1, d2, d3, d4, q1, q2;

  top = floorf(h); bottom = ceilf(h);
  left = floorf(w); right = ceilf(w);

  d1 = w - left;
  d2 = right - w;
  d3 = h - top;
  d4 = bottom - h;

  q1 = d4*get_pixel(im, c, top, left) + d3*get_pixel(im, c, bottom, left);
  q2 = d4*get_pixel(im, c, top, right) + d3*get_pixel(im, c, bottom, right);

  return d2*q1 + d1*q2;
}

image bilinear_resize(image im, int h, int w)
{
  int k, j, i;
  float x, y, ax, ay, bx, by;
  image resized = make_image(im.c, h, w);
    
  ax = (float) im.w / w;
  ay = (float) im.h / h;        
  bx = 0.5 * (ax - 1.0);
  by = 0.5 * (ay - 1.0);
  for (k = 0; k < im.c; k++) {
    for (j = 0; j < h; j++) {
      y = ay * j + by; 
      for (i = 0; i < w; i++) {
        x = ax * i + bx;
        set_pixel(resized, k, j, i, bilinear_interpolate(im, k, y, x));
      }
    }
  }
  return resized;
}
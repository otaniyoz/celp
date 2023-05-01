#include "args.h"
#include "image.h"
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
  if (0 == strcmp(argv[1], "help")) {
    printf("very helpful message will go here\n");
  } else if (0 == strcmp(argv[1], "resize")) {
    image im = load_image(argv[2]);
    save_image(bilinear_resize(im, 1080.0*im.h/im.w, 1080), argv[3]);
  } else if (0 == strcmp(argv[1], "blur")) {
    image im = load_image(argv[2]);
    save_image(smooth_image(im, 2.4), argv[3]);
    free_image(im);
  } else if (0 == strcmp(argv[1], "otsu")) {
    image im = load_image(argv[2]);
    im = rgb_to_grayscale(im);
    apply_otsu_threshold(im, 256);
    save_image(im, argv[3]);
    free_image(im);
  } else if (0 == strcmp(argv[1], "diffuse-dither")) {
    image im = load_image(argv[2]);
    image q = load_image(argv[3]);
    floyd_steinberg_dither(im, q);
    save_image(im, argv[4]);
    free_image(im);
    free_image(q);
  } else if (0 == strcmp(argv[1], "colorize-sobel")) {
    image im = load_image(argv[2]);
    save_image(colorize_sobel(im), argv[3]);
    free_image(im);
  } else if (0 == strcmp(argv[1], "canny")) {
    image im = load_image(argv[2]);
    save_image(get_canny_edges(im, 0.4, 0.01, 0.25), argv[3]);
    free_image(im);
  } else if (0 == strcmp(argv[1], "xdog")) {
    image im = load_image(argv[2]);
    image S = get_xdog_edges(im, 0.6, 0, 0, 0, 8.5, 180, 0.25, 180, 0);
    //image S = get_xdog_edges(im, 2.0, 0.1, 20, 7.2, 8.5, 40, 1.0, 0.01, 1);
    //image S = get_xdog_edges(im, 1.4, 3.76, 2.2, 2.4, 18.5, 21.7, 0.8, 0.5, 1);
    S = grayscale_to_rgb(S, 1, 1, 1);
    save_image(S, argv[3]);
    free_image(im);
    free_image(S);
  } else if (0 == strcmp(argv[1], "sort")) {
    image im = load_image(argv[2]);
    sort_pixels(im, 0.35, 0.25, 1, 1);
    save_image(im, argv[3]);
  }
  return 0;
}

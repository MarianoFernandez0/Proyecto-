function [g] = draw_circle(x0, y0, r, cc)
% x,y coordinates
% r radius

x = x0 - r;
y = y0 - r;
w = 2*r;
h = 2*r;

g = rectangle('Position', [x,y,w,h], 'Curvature', [1 1], 'EdgeColor', cc);

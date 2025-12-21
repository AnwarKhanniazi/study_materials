%%
a = imread('pk.jpg');
b=rgb2gray(a);
figure(1)
imshow(b);
axis on;
title("original image");
%% filtering concept
h=[1 1 1; 1 1 1;1 1 1]/9;
h1=[0 1 0; 1 1 1; 0 1 0]/5;
h2=[0 0 0; 0 1 0;0 0 0];
h3=[0 0 0;0 0 1;0 0 0];
h4=[1 0 -1;2 0 -2;1 0 -1];
h5=[1 2 1;2 4 2;1 2 1]/16;
c = imfilter(b,h);
figure(2)
imshow(c);
title("smoth filtering");
axis on;
d = imfilter(b,h1);
figure(3)
imshow(d);
title("smoth filtering");
axis on;
e = imfilter(b,h2);
figure(4)
imshow(e);
title("smoth filtering");
axis on;
f = imfilter(b,h3);
figure(5)
imshow(f);
title("smoth filtering");
axis on;
g = imfilter(b,h4);
figure(6)
imshow(g);
title("edge detection");
axis on;
h=edge(b);
figure(7)
imshow(h);
title("edge detection");
axis on;

i=edge(b,'canny');
figure(8)
imshow(i);
title("edge detection");
axis on;

j=edge(b,'prewitt');
figure(9)
imshow(j);
title("edge detection");
axis on;
k=bwmorph(j,'dilate',2);
figure(10)
imshow(k)
title("edge detection");
axis on;

l = imfilter(b,h5);
figure(11)
imshow(l);
title("gussian filtering");
axis on;



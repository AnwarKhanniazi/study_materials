 
close all;
clear all;
clc;
 
I = imread('pk.jpg');
Im = rgb2gray(I);
 
noisy = imnoise(Im, 'salt & pepper',0.1);
 
[m,n] = size(noisy);
 
output = zeros(m,n);
output = uint8(output);
 
for i = 1:m
    for j = 1:n  %intesity of pixel in the noisy image is given as noisy(i,j)
        % here we define max and minimum values x and y coordinates of any
        % pixel can take
        xmin = max(1,i-1); % minimum x coordinate has to be greater than or equal to 1
        xmax = min(m,i+1);
        ymin = max(1,j-1);
        ymax = min(n,j+1);
        % the neighbourhood matrix will then be
        temp = noisy(xmin:xmax, ymin:ymax);
        %now the new intensity of pixel at (i,j) will be median of this
        %matrix
        output(i,j) = median(temp(:));
    end
end
 
figure(1);
set(gcf, 'Position', get(0,'ScreenSize'));
subplot(131),imshow(I),title('Original Image');
subplot(132),imshow(noisy),title('noisy Image');
subplot(133),imshow(output),title('output of median filter');
I = imread('pk.jpg');
N=imnoise(I,'salt & pepper', 0.03);
mf = ones(3, 3)/9;
noise_free = imfilter(N,mf);

subplot(2,2,1),imshow(I), title('Original Image');
axis on;
subplot(2,2,2),imshow(N), title('Noisy Image');
axis on;
subplot(2,2,3),imshow(noise_free), title('After Removing Noise');
axis on;
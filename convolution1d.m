
%% Example 1
x=[2 1 2 1]
y=[1 2 3]
conv(x,y);


x=[1 1 1 1 1];
h=[0 1 2 3];
y=conv(x,h);
subplot(313),stem(y);
title("convolution 1d signal ");
subplot(311),stem(x);
title(" x vector signal ");
subplot(312),stem(h)
title("h vector signal");
n=0:40;
x=sin(.4*n);
h=sin(.8*n);
y=conv(x,h)
subplot(313),stem(n,y(1:length(n)));
title("convolution 1d signal ");
subplot(311),stem(n,x(1:length(n)));
title(" x vector signal ");
subplot(312),stem(n,h(1:length(n)));
title("h vector signal"); 

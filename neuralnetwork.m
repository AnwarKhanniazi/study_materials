clear all;
clc;
X=[
    0.5 0.7 0.2;
    0.4 0.1 0.5
  ]
Y=[0.6 0.3 0.4]
%all parms w1 on layer1 and w2 on layer2
W1=[
  0.2 0.15;
  0.1 0.2;
]
J=[
   0.5;
   0.8;
]
B=[
  0.2;
  0.1;
]
W2=[
    0.3 0.7;
]
%Info: setting num_of_iterations=75 and learning_rate=0.9,(loss=0.0373)
%better performance while on setting 0.5 and 3 iteration loss reduces
num_of_iterations=3;%Sir kindly set 75 for better performance
learning_rate=0.5;
fprintf('Normalized X at start');
X=Normalization(X)
fprintf('\nFirst time out of iteration feeding or forward propagation..\n');
[y_predicted,a_layer1,z_norm_layer1,z_hat_layer1,unitvariance]=feed_forward(X,W1,W2,J,B);
loss1=y_predicted-Y;
loss1
loss=sumsqr(loss1)
y_predicted
fprintf('Iterating..Gradient and feeding..\n');
for i=1:num_of_iterations
fprintf('Iteration %d',i);
[X,W1,W2,J,B]=Gradient_descent(X,W1,W2,J,B,y_predicted,a_layer1,Y,learning_rate,z_norm_layer1,z_hat_layer1,unitvariance)
[y_predicted,a_layer1,z_norm_layer1,z_hat_layer1,unitvariance]=feed_forward(X,W1,W2,J,B);
loss=y_predicted-Y;
y_predicted
loss=sumsqr(loss)
end
fprintf('Name : Anwar Khan.\nID : SP20-RCS-008\n');
%fprintf('Info: setting num_of_iterations=75 and \n learning_rate=0.5,(loss=0.0373) better performance \n while on setting learning_rate=0.5 and 3 iteration loss reduces ');
 
 
function [X,W1,W2,J,B]=Gradient_descent(X,W1,W2,J,B,y_predicted,a_layer1,Y,learning_rate,z_norm_layer1,z_hat_layer1,unitvariance)
%params to learn
W1_11=W1(1,1); 
W1_12=W1(1,2);
W1_21=W1(2,1);
W1_22=W1(2,2);
 
W2_11=W2(1,1);
W2_12=W2(1,2);
 
J11=J(1,1);
J21=J(2,1);
 
B11=B(1,1);
B21=B(2,1);
 
 
 
%derivation
dJ_wrt_dy_hat=2 * (y_predicted-Y);
dJ_wrt_dy_hat
dy_hat_wrt_W2_11=a_layer1(1,:);
dy_hat_wrt_W2_11
dy_hat_wrt_W2_12=a_layer1(2,:);
 
dy_hat_wrt_W2_12
 
dy_hat_wrt_a_layer1_1=W2_11;
dy_hat_wrt_a_layer1_1
dy_hat_wrt_a_layer1_2=W2_12;
dy_hat_wrt_a_layer1_2
 
da_layer1_wrt_z_hat=z_hat_layer1;
da_layer1_wrt_z_hat
da_layer1_wrt_z_hat(da_layer1_wrt_z_hat>=0)=1;
 
da_layer1_wrt_z_hat(da_layer1_wrt_z_hat<0)=0;
da_layer1_wrt_z_hat
dz_hat1_1_wrt_j1_1=z_norm_layer1(1,:);
dz_hat1_1_wrt_j1_1
dz_hat1_2_wrt_j2_1=z_norm_layer1(2,:);
dz_hat1_2_wrt_j2_1
dz_hat1_1_wrt_B1_1=1;
 
dz_hat1_2_wrt_B2_1=1;
 
dz_hat1_1_wrt_z_norm1_1=J11;
dz_hat1_1_wrt_z_norm1_1
dz_hat1_2_wrt_z_norm1_2=J21;
dz_hat1_2_wrt_z_norm1_2
 
dz_norm1_1_wrt_z_layer1_1=(1./sqrt(unitvariance+0.0001));
sqrt(unitvariance+0.0001)
dz_norm1_1_wrt_z_layer1_1
dz_norm1_1_wrt_z_layer1_2=(1./sqrt(unitvariance+0.0001));
dz_norm1_1_wrt_z_layer1_2
 
dz_norm1_1_wrt_z_layer1_1=dz_norm1_1_wrt_z_layer1_1(1,:);
dz_norm1_1_wrt_z_layer1_1
dz_norm1_1_wrt_z_layer1_2=dz_norm1_1_wrt_z_layer1_2(2,:);
dz_norm1_1_wrt_z_layer1_2
X(1,:)
 
 
%update weights
W2_11=W2_11-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_W2_11);
W2_12=W2_12-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_W2_12);
 
J11=J11-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_1 .* da_layer1_wrt_z_hat(1,:) .* dz_hat1_1_wrt_j1_1);
J21=J21-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_2 .* da_layer1_wrt_z_hat(2,:) .* dz_hat1_2_wrt_j2_1);
 
B11=B11-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_1 .* da_layer1_wrt_z_hat(1,:) .* dz_hat1_1_wrt_B1_1);
B21=B21-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_2 .* da_layer1_wrt_z_hat(2,:) .* dz_hat1_2_wrt_B2_1);
 
 
W1_11=W1_11-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_1 .* da_layer1_wrt_z_hat(1,:) .* dz_hat1_1_wrt_z_norm1_1 .* dz_norm1_1_wrt_z_layer1_1 .* X(1,:));
W1_12=W1_12-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_1 .* da_layer1_wrt_z_hat(1,:) .* dz_hat1_1_wrt_z_norm1_1 .* dz_norm1_1_wrt_z_layer1_1 .* X(2,:));
 
W1_21=W1_21-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_2 .* da_layer1_wrt_z_hat(2,:) .* dz_hat1_2_wrt_z_norm1_2 .* dz_norm1_1_wrt_z_layer1_2 .* X(1,:));
W1_22=W1_22-learning_rate*sum(dJ_wrt_dy_hat .* dy_hat_wrt_a_layer1_2 .* da_layer1_wrt_z_hat(2,:) .* dz_hat1_2_wrt_z_norm1_2 .* dz_norm1_1_wrt_z_layer1_2 .* X(2,:));
 
 
 
 
 
%updated parms
 
W2(1,1)=W2_11;
W2(1,2)=W2_12;
 
J(1,1)=J11;
J(2,1)=J21;
 
B(1,1)=B11;
B(2,1)=B21;
 
W1(1,1)=W1_11;
W1(1,2)=W1_12;
W1(2,1)=W1_21;
W1(2,2)=W1_22;
 
%y_predicted,a_layer2=feed_forward(X,W1,W2,J,B);
 
 
 
end
 
 
function [y_predicted,a_layer1,z_norm_layer1,z_hat_layer1,unitvariance]=feed_forward(X,W1,W2,J,B)
[a_layer1,z_norm_layer1,z_hat_layer1,unitvariance]=layer1_feed(W1,X,J,B);
y_predicted=layer2_feed(a_layer1,W2);
end
function [a_output_layer1,z_norm_layer1,z_hat_layer1,unitvariance]=layer1_feed(W1,X,J,B)
z_layer1=W1 * X;
z_layer1
[z_norm_layer1,unitvariance]=Normalization(z_layer1);
z_norm_layer1
J(1,1)
z_norm_layer1(1,:)
z_hat_layer1=[J(1,1)*z_norm_layer1(1,:)+B(1,1);J(2,1)*z_norm_layer1(2,:)+B(2,1)];
z_hat_layer1
a_output_layer1=z_hat_layer1;
 
a_output_layer1(a_output_layer1<0)=0;%Relu function
end
function y_predicted=layer2_feed(a_layer1,W2)
a_layer1
y_predicted= W2 * a_layer1;
y_predicted
end
function [output,unitvariance]=Normalization(Xv)
% zeromean=(Xv-mean(Xv(:)));%zero mean
% unitvariance=(zeromean/std(zeromean(:)));%unit variance
%reference https://stackoverflow.com/questions/8717139/how-to-normalize-a-signal-to-zero-mean-and-unit-variance
%according to class I have done it for 3 examples.
Xv
U=[sum(Xv(1,:))/3;sum(Xv(2,:))/3;];
 
zeromean=Xv-U;
zeromean
unitvariance=(sum((zeromean).^2,2)./3);
tmp1=(Xv-U);
epsilon=0.0001;
%unitvariance
tmp2=sqrt(unitvariance+epsilon);
%tmp2
output=tmp1./tmp2;
%output
%output=unitvariance;
end
 
 


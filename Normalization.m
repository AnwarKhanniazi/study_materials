function [output,unitvariance]=Normalization(Xv)
% zeromean=(Xv-mean(Xv(:)));%zero mean
% unitvariance=(zeromean/std(zeromean(:)));%unit variance
%reference https://stackoverflow.com/questions/8717139/how-to-normalize-a-signal-to-zero-mean-and-unit-variance
%according to class I have done it for 3 examples.
U=[sum(Xv(1,:))/3;sum(Xv(2,:))/3;];
zeromean=Xv-U;
unitvariance=(((zeromean).^2)./3);
tmp1=(Xv-U);
epsilon=0.0001;
tmp2=sqrt(unitvariance+epsilon);
output=tmp1./tmp2;
%output=unitvariance;
end
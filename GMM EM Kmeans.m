clc;
clear all;

%% K-Means implementation
mu1=[-1 -1];
mu2=[1 1];
sigma1=[.8 0;0 .8];
sigma2=[.75 -0.2;-0.2 .6];
x=rand (1000,1);
X=zeros(1000,2);
for i=1:1000
   if x(i)<=.7
       X(i,1:2)=mvnrnd(mu1,sigma1,1);
       continue
   end
  X(i,1:2)=mvnrnd(mu2,sigma2,1);
end
a=-5:0.01:5;
b=-5:0.01:5;
figure
hold on
[A,B]=meshgrid(a,b);
X1=mvnpdf([A(:),B(:)],mu1,sigma1);
X1=reshape(X1,size(A));
X2=mvnpdf([A(:),B(:)],mu2,sigma2);
X2=reshape(X2,size(A));
Xtotal=0.7*X1+0.3*X2;
contour(A,B,Xtotal);
scatter(X(:,1),X(:,2),10,'.');
title ('K- Means implementation with 2 Gaussians');
xlabel ('x');
ylabel('y');
hold off
%% EM impl  ementation for 2 gaussians
X=zeros(10000,2);
x=rand(10000,1);
for i=1:10000
   if x(i)<=0.7
       X(i,:)=mvnrnd(mu1,sigma1,1);
       continue
   end
  X(i,:)=mvnrnd(mu2,sigma2,1);
end
w=rand(10000,1);
w=cat(2,w,(ones(10000,1)-w));
mu_1=rand(1,2);
mu_2=rand(1,2);
sigma_1=-1;
sigma_2=-1;
while (1)
    sigma_1=rand(2,2);
    sigma_2=rand(2,2);
    if (det(sigma_1)>0 && det(sigma_2)>0)
        break
    end
end
phi_1=rand(1,1);
phi_2=1-phi_1;
new=ones(7,2);
check=1;
k=1;
while(check>10^-4)
    last=new;
    loss(k)=0;
   for i=1:10000
     N1=exp(-0.5*((X(i,:)-mu_1)*inv(sigma_1)*(X(i,:)-mu_1)'))/sqrt((2*pi)^2*abs(det(sigma_1)));
     N2=exp(-0.5*((X(i,:)-mu_2)*inv(sigma_2)*(X(i,:)-mu_2)'))/sqrt((2*pi)^2*abs(det(sigma_2)));
     w(i,1)=phi_1*N1/(phi_1*N1+phi_2*N2);   
     w(i,2)=phi_2*N2/(phi_1*N1+phi_2*N2);
     loss(k)=loss(k)+log(phi_1*N1+phi_2*N2);
   end
    phi_1=sum(w(:,1))/10000;
    phi_2=sum(w(:,2))/10000;
    mu_1=zeros(1,2);
    mu_2=zeros(1,2);
    for i=1:10000
        mu_1=mu_1+w(i,1).*X(i,:);
        mu_2=mu_2+w(i,2).*X(i,:);
    end
   
    sum_w=sum(w);
        mu_1=mu_1./sum_w(1);
        mu_2=mu_2./sum_w(2);
        sigma_1=zeros(2,2);
        sigma_2=zeros(2,2);
        for i=1:10000
            sigma_1=sigma_1+w(i,1).*((X(i,:)-mu_1)'*(X(i,:)-mu_1));
            sigma_2=sigma_2+w(i,2).*((X(i,:)-mu_2)'*(X(i,:)-mu_2));
        end
        sigma_1=sigma_1./sum_w(1);
        sigma_2=sigma_2./sum_w(2);
        new=cat(1,mu_1,mu_2);
        new=cat(1,new,sigma_1);
        new=cat(1,new,sigma_2);
        temp=cat(2,phi_1,phi_2);
        new=cat(1,new,temp);
        check=norm(new-last);
        k=k+1;
end
figure
plot(1:k-1,loss);
xlabel('# of Iteration');
ylabel('log-likelihood value');
title ('log-likelihood function for 2 gaussians');
a=-5:0.01:5;
b=-5:0.01:5;
figure
hold on
[A,B]=meshgrid(a,b);
Z1=mvnpdf([A(:),B(:)],mu_1,sigma_1);
Z1=reshape(Z1,size(A));
Z2=mvnpdf([A(:),B(:)],mu_2,sigma_2);
Z2=reshape(Z2,size(A));
Ztotal=Z1.*phi_1+Z2.*phi_2;
contour(A,B,Ztotal);
scatter(X(:,1),X(:,2),2.5,'.');
title ('EM implementation for 2 gaussians');
xlabel ('x');
ylabel('y');
%% EM implementation for 3 gaussians

clear loss
w=0.5.*rand(10000,1);
w=cat(2,w,0.5.*rand(10000,1));
temp=w(1:10000,1)+w(1:10000,2);
w=cat(2,w,ones(10000,1)-temp);
mu_1=rand(1,2);
mu_2=rand(1,2);
mu_3=rand(1,2);

while (1)
    sigma_1=rand(2,2);
    sigma_2=rand(2,2);
    sigma_3=rand(2,2);
    if (det(sigma_1)>0 && det(sigma_2)>0 && det(sigma_3)>0)
        break
    end
end


sigma_1=rand(2,2);
sigma_2=rand(2,2);
sigma_3=rand(2,2);
phi_1=0.5*rand(1,1);
phi_2=0.5*rand(1,1);
phi_3=1-phi_1-phi_2;
new=ones(1,2);
check=1;
k=1;
while(check>10^-3)
    last=new;
    loss(k)=0;  
   for i=1:10000
     N1=exp(-0.5*((X(i,1:2)-mu_1)*inv(sigma_1)*(X(i,1:2)-mu_1)'))/sqrt((2*pi)^2*abs(det(sigma_1)));
     N2=exp(-0.5*((X(i,1:2)-mu_2)*inv(sigma_2)*(X(i,1:2)-mu_2)'))/sqrt((2*pi)^2*abs(det(sigma_2)));
     N3=exp(-0.5*((X(i,1:2)-mu_3)*inv(sigma_3)*(X(i,1:2)-mu_3)'))/sqrt((2*pi)^2*abs(det(sigma_3)));
     w(i,1)=phi_1*N1/(phi_1*N1+phi_2*N2+phi_3*N3);   
     w(i,2)=phi_2*N2/(phi_1*N1+phi_2*N2+phi_3*N3);
     w(i,3)=phi_3*N3/(phi_1*N1+phi_2*N2+phi_3*N3);
     loss(k)=loss(k)+log(phi_1*N1+phi_2*N2+phi_3*N3);
   end
    phi_1=sum(w(1:10000,1))/10000;
    phi_2=sum(w(1:10000,2))/10000;
    phi_3=sum(w(1:10000,3))/10000;
    mu_1=zeros(1,2);
    mu_2=zeros(1,2);
    mu_3=zeros(1,2);
    for i=1:10000
        mu_1=mu_1+w(i,1).*X(i,1:2);
        mu_2=mu_2+w(i,2).*X(i,1:2);
        mu_3=mu_3+w(i,3).*X(i,1:2);
    end
    sum_w=sum(w);
    mu_1=mu_1./sum_w(1);
    mu_2=mu_2./sum_w(2);
    mu_3=mu_3./sum_w(3);
    sigma_1=zeros(2,2);
    sigma_2=zeros(2,2);
    sigma_3=zeros(2,2);
    for i=1:10000
        sigma_1=sigma_1+w(i,1).*((X(i,1:2)-mu_1)'*(X(i,1:2)-mu_1));
        sigma_2=sigma_2+w(i,2).*((X(i,1:2)-mu_2)'*(X(i,1:2)-mu_2));
        sigma_3=sigma_3+w(i,3).*((X(i,1:2)-mu_3)'*(X(i,1:2)-mu_3));
    end
    sigma_1=sigma_1./sum_w(1);
    sigma_2=sigma_2./sum_w(2);
    sigma_3=sigma_3./sum_w(3);
    new=cat(1,mu_1,mu_2);
    new=cat(1,new,mu_3);
    new=cat(1,new,sigma_1);
    new=cat(1,new,sigma_2);
    new=cat(1,new,sigma_3);
    temp=cat(2,phi_1,phi_2);
    new=cat(1,new,temp);
    temp=cat(2,phi_3,zeros(1,1));
    new=cat(1,new,temp);
    check=norm(new-last);
    k=k+1;
end
figure
plot(1:k-1,loss);
xlabel('# of Iteration');
ylabel('log-likelihood value');
title ('log-likelihood function for 3 gaussians');
a=-5:0.01:5;
b=-5:0.01:5;
[A,B]=meshgrid(a,b);
Z1=mvnpdf([A(:),B(:)],mu_1,sigma_1);
Z1=reshape(Z1,size(A));
Z2=mvnpdf([A(:),B(:)],mu_2,sigma_2);
Z2=reshape(Z2,size(A));
Z3=mvnpdf([A(:),B(:)],mu_3,sigma_3);
Z3=reshape(Z3,size(A));
Z4=Z1*phi_1+Z2*phi_2+Z3*phi_3;
figure
hold on
contour(A,B,Z4);
scatter(X(:,1),X(:,2),2.5,'.');
title ('EM implementation for 3 gaussians');
xlabel ('x');
ylabel('y');
hold off


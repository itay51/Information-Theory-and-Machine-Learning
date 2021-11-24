% =========================================
%       Homework on K-Nearest Neighbors
% =========================================
% Course: Introduction to Information Theory
% Lecturer: Haim H. Permuter.
%
% NOTE:
% -----
% Please change the variable ID below to your ID number as a string.
% Please do it now and save this file before doing the assignment
%  clear all;
ID = '308574656';

%% Loading and plot a sample of the data
% ---------------------------------------

load('MNIST_3_and_5.mat')
%% Find optimal d and k
% ---------------------------------------
% for p=1:5
%     d=zeros(length(Xtrain(:,1)),length(Xvalid(:,1)));
%     for i=1:length(Xtrain(:,1))
%         for j=1:length(Xvalid(:,1))
%             d(i,j)=norm(Xtrain(i,:)-Xvalid(j,:),p);
%         end
%     end
%     for k=1:2:27
% 
%         closeValue=zeros(k,length(Xvalid(:,1)));
%         closeIndex=zeros(k,length(Xvalid(:,1)));
%         temp=d;
%         closeNum=zeros(k,length(Xvalid(:,1)));
% 
%         for j=1:length(Xvalid(:,1))
%             temp=d;
%             for i=1:k
%                 [closeValue(i,j), closeIndex(i,j)]=min(temp(:,j));
%                 temp(closeIndex(i,j),j)=100000;
%                 closeNum(i,j)=Ytrain(closeIndex(i,j));
% 
% 
%             end
% 
%         end
%         Y=mean(closeNum,1);
%         Y(Y>4)=5;
%         Y(Y<4)=3;
%         error(p,(k+1)/2)=(sum(Y'~=Yvalid));
%     end
% end


%% find Ytest
% --------------------------------------------
%the optimal error is obtained by choosing norm=5 and k=23
p=5;
k=23;
d=zeros(length(Xtrain(:,1)),length(Xtest(:,1)));
    for i=1:length(Xtrain(:,1))
        for j=1:length(Xtest(:,1))
            d(i,j)=norm(Xtrain(i,:)-Xtest(j,:),p);
        end
    end
        closeValue=zeros(k,length(Xtest(:,1)));
        closeIndex=zeros(k,length(Xtest(:,1)));
        temp=d;
            closeNum=zeros(k,length(Xtest(:,1)));

 for j=1:length(Xtest(:,1))
            temp=d;
            for i=1:k
                [closeValue(i,j), closeIndex(i,j)]=min(temp(:,j));
                temp(closeIndex(i,j),j)=100000;
                closeNum(i,j)=Ytrain(closeIndex(i,j));


            end

 end
        Y=mean(closeNum,1);
        Y(Y>4)=5;
        Y(Y<4)=3;
        Ytest=Y';

%save classification results

disp('saving')
csvwrite([ID '.txt'], Ytest)
disp('done')
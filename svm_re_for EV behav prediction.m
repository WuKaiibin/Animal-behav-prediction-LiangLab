%% Initialization
warning off             % Disable warnings
close all               % Close all figures
clear                   % Clear workspace
clc                     % Clear command window

%% Data setup
result1 = [];

%%
times=[];rmse_sum=[];num1 =5;
cishu = 10;roi_num = 148;
pt = floor(roi_num*0.9);
time_num = 301;
for j = 1:cishu
ab =[rand(roi_num-num1,1) result1(num1+1:roi_num,:)];
Aaa = sortrows(ab);
A = [Aaa(:,2:end);result1(1:num1,:)];


% Aaa = result1(num1+1:roi_num,:);
% A = [Aaa(:,1:end);result1(1:num1,:)];


for i = 1:time_num-1

%% Prepare dataset
result=A(:,1:end);
num_samples = length(result);  %  
% kim = 10;                      % time window size
% zim =  10;                      % prediction offset
 
%% Sequence generation (commented out)
% for i = 1: num_samples - kim - zim + 1
%     res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
% end
%  
% %%  
% temp = 1: 1: 922;
% 
% P_train = res(temp(1: 700), 1: 15)';
% T_train = res(temp(1: 700), 16)';
% M = size(P_train, 2);
% 
% P_test = res(temp(701: end), 1: 15)';
% T_test = res(temp(701: end), 16)';
% N = size(P_test, 2);
%% Parameter settings
% outdim = 1;                                  % 
% num_size = 0.7;                              % 
% num_train_s = round(num_size * num_samples); % 
% f_ = size(res, 2) - outdim;                  % 
 
%% Alternative training set (commented)
% P_train = [result(1:60 , 1:i);result(72:72+60 , 1:i)]';
% T_train = [result(1:60 , 176);result(72:72+60 , 176)]';
% M = size(P_train, 2);
%  
% P_test = [result(61:71 , 1:i);result(72+61:end-1 , 1:i)]';
% T_test = [result(61:71 , 176);result(72+61:end-1 , 176)]';
% N = size(P_test, 2);
 
P_train = [result(1:pt , 1:i)]';
T_train = [result(1:pt , time_num)]';
M = size(P_train, 2);
 
P_test = [result(pt+1:roi_num , 1:i)]';
T_test = [result(pt+1:roi_num , time_num)]';
N = size(P_test, 2);


%% Normalize data
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
 
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
 
%% Filter dataset
kkk=1;p_train1=[];t_train1=[];
for ji = 1:length(p_train(1,:))

    if p_train(1,ji)<0.6 && p_train(1,ji)>0.3

        p_train1(:,kkk) =  p_train(:,ji);
        t_train1(1,kkk) = t_train(1,ji);
        kkk=kkk+1;
    end
end

kkk=1;p_test1=[];t_test1=[];
for ji = 1:length(p_test(1,:))

    if p_test(1,ji)<0.6 && p_test(1,ji)>0.3

        p_test1(:,kkk) =  p_test(:,ji);
        t_test1(1,kkk) = t_test(1,ji);
        kkk=kkk+1;
    end
end


p_train = p_train1'; p_test = p_test1';

% p_train = P_train'; p_test = P_test';

t_train = t_train1'; t_test = t_test1';
 



%% Train model
% c = 4.0;    % 
% g = 0.8;    % 
% cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
% model = svmtrain(t_train, p_train, cmd);,'auto'
  model = fitcsvm(p_train, t_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.05,...
    'KernelFunction','polynomial');     %5%
%% Evaluate model
[t_sim1, error_1] = predict(model, p_train);
[t_sim2, error_2] = predict(model , p_test );
 kk=1;tt=[];tt_out=[];
for ii =1:length(t_sim2)
    if t_test(ii) == 1
        tt(kk) = t_sim2(ii);
        tt_out(kk) = t_test(ii);
kk=kk+1;
    end
end


%% Post-processing
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
 TT=mapminmax('reverse', tt, ps_output);
 T_sim2=1.-T_sim2;
%% Adjust predictions
TT=1.-TT;
times(j,i) = mean(TT);

%% Calculate error
error=T_sim2'-t_test1;
[c,l]=size(t_test1);
MAE1=sum(abs(error))/l;
MSE1=error*error'/l;
RMSE1=MSE1^(1/2);
rmse_sum(j,i) = RMSE1;

%% Calculate subset error
error=TT-tt_out;
[c,l]=size(tt_out);
MAE1=sum(abs(error))/l;
MSE1=error*error'/l;
RMSE1=MSE1^(1/2);
rmse_sum_1(j,i) = RMSE1;


end

end

%% Plot results
times1=mean(times);
% rmse_sum1=mean(rmse_sum);
% % rmse_sum_1_1 = mean(rmse_sum_1);
% P1= polyfit(1:time_num-1, times1, 5) ;
% yi= polyval(P1, 1:time_num-1);
figure
% plot(1:175, yi, 'g', 'LineWidth', 1)
hold on 
plot(1:time_num-1,times1,'bo')

xlabel('Prediction')
ylabel('Prediction')
% string = {''; ['RMSE=' num2str(error1)]};
title(string)
xlim([1,time_num-1])
ylim([0,1]);
grid

%% Permutation

we=result1(:,time_num);
for jii = 1:10000

    dddd = rand(roi_num,1);
    td = [dddd we];
    tdd =sortrows(td);
    we = tdd(:,2);
end
%% mesrRUC
rmse_sum1=mean(rmse_sum);
% rmse_sum_1_1 = mean(rmse_sum_1);
x = 50:175;
acruc(1) = mean(rmse_sum1(x));

% x = 126:175;
% acruc(2) = mean(rmse_sum1(x));

% x = 126:175;
% acruc(3) = mean(rmse_sum1(x));

%% Summary statistics
rmse_sum1=mean(rmse_sum);

x = 1:50;
acruc_ctrl(1) = mean(rmse_sum1(x));

x = 51:125;
acruc_ctrl(2) = mean(rmse_sum1(x));

x = 126:175;
acruc_ctrl(3) = mean(rmse_sum1(x));









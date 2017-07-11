function CdfErrors(data1, data2)
%CREATEFIGURE(data1, Y1, data2, Y2)
%  data1:  x 데이터의 벡터
%  data2:  x 데이터의 벡터

%  MATLAB에서 19-Jun-2017 02:36:54에 자동 생성됨

% figure 생성
figure1 = figure;

% axes 생성
axes1 = axes('Parent',figure1);
hold(axes1,'on');

hold(axes1,'on');
% plot 생성
cdfplot(abs(data1));

% plot 생성
cdfplot(abs(data2));

% title 생성
title('Empirical CDF','FontWeight','bold');

box(axes1,'on');
grid(axes1,'on');
% 나머지 axes 속성 설정
set(axes1,'FontSize',14,'FontWeight','bold','XTick',...
    [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]);
% legend 생성
legend1 = legend(axes1,'show');
set(legend1,'FontSize',24);


function CdfErrors(data1, data2)
%CREATEFIGURE(data1, Y1, data2, Y2)
%  data1:  x �������� ����
%  data2:  x �������� ����

%  MATLAB���� 19-Jun-2017 02:36:54�� �ڵ� ������

% figure ����
figure1 = figure;

% axes ����
axes1 = axes('Parent',figure1);
hold(axes1,'on');

hold(axes1,'on');
% plot ����
cdfplot(abs(data1));

% plot ����
cdfplot(abs(data2));

% title ����
title('Empirical CDF','FontWeight','bold');

box(axes1,'on');
grid(axes1,'on');
% ������ axes �Ӽ� ����
set(axes1,'FontSize',14,'FontWeight','bold','XTick',...
    [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]);
% legend ����
legend1 = legend(axes1,'show');
set(legend1,'FontSize',24);


function HistErrors(data1, data2)
%CREATEFIGURE(DATA1, DATA2)
%  DATA1:  histogram data
%  DATA2:  histogram data

%  MATLAB���� 19-Jun-2017 02:03:13�� �ڵ� ������

% figure ����
figure1 = figure;

% axes ����
axes1 = axes('Parent',figure1);
hold(axes1,'on');

hold(axes1,'on');
% histogram ����
histogram(data1,'DisplayName','Systolic','Parent',axes1,'NumBins',100);

% histogram ����
histogram(data2,'DisplayName','Diastolic','Parent',axes1,'NumBins',100);

% ���� ������ �ּ� ó���� �����Ͽ� ��ǥ���� X ������ ����
xlim(axes1,[-30 30]);
% ���� ������ �ּ� ó���� �����Ͽ� ��ǥ���� Y ������ ����
ylim(axes1,[0 10200]);
% ������ axes �Ӽ� ����
set(axes1,'FontSize',14,'FontWeight','bold','XGrid','on','XMinorTick','on',...
    'XTick',[-40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40],'YGrid',...
    'on');
% legend ����
legend1 = legend(axes1,'show');
set(legend1,'FontSize',24);


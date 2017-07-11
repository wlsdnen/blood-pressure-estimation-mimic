function HistErrors(data1, data2)
%CREATEFIGURE(DATA1, DATA2)
%  DATA1:  histogram data
%  DATA2:  histogram data

%  MATLAB에서 19-Jun-2017 02:03:13에 자동 생성됨

% figure 생성
figure1 = figure;

% axes 생성
axes1 = axes('Parent',figure1);
hold(axes1,'on');

hold(axes1,'on');
% histogram 생성
histogram(data1,'DisplayName','Systolic','Parent',axes1,'NumBins',100);

% histogram 생성
histogram(data2,'DisplayName','Diastolic','Parent',axes1,'NumBins',100);

% 다음 라인의 주석 처리를 제거하여 좌표축의 X 제한을 유지
xlim(axes1,[-30 30]);
% 다음 라인의 주석 처리를 제거하여 좌표축의 Y 제한을 유지
ylim(axes1,[0 10200]);
% 나머지 axes 속성 설정
set(axes1,'FontSize',14,'FontWeight','bold','XGrid','on','XMinorTick','on',...
    'XTick',[-40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40],'YGrid',...
    'on');
% legend 생성
legend1 = legend(axes1,'show');
set(legend1,'FontSize',24);


function [ output_args ] = SaveError( arr, pic_range, error_range )
%SAVEERROR 이 함수의 요약 설명 위치
%   자세한 설명 위치

PPG_IDX = 1;
BP_IDX  = 2;
ECG_IDX = 3;

for i = 1:size(arr, 1)
   
    h=figure;
    
    min_idx = max (1, arr(i,4)-pic_range);
    max_idx = min (length (Part_1{arr(i,3)}), arr(i,4)+pic_range);
    
    subplot (3,1,1);
    
    plot (Part_1{arr(i,3)}(BP_IDX, min_idx : max_idx));
    xlim([0 2*pic_range])
    hold on;
    plot (pic_range, Part_1{arr(i,3)}(BP_IDX, arr(i,4)), 'Marker', 'o');
    title ('Blood Pressure');
    hold off;
    
    subplot (3,1,2);
    plot (Part_1{arr(i,3)}(PPG_IDX, min_idx : max_idx));
    xlim([0 2*pic_range])
    hold on;
    plot (pic_range, Part_1{arr(i,3)}(PPG_IDX, arr(i,4)), 'Marker', 'o');
    title ('PPG');
    hold off;
    
    subplot (3,1,3);
    plot (Part_1{arr(i,3)}(ECG_IDX, min_idx : max_idx));
    xlim([0 2*pic_range])
    hold on;
    plot (pic_range, Part_1{arr(i,3)}(ECG_IDX, arr(i,4)), 'Marker', 'o');
    title ('ECG');
    hold off;
    
    suptitle(sprintf ('Error - SBP : %03.4f, DBP : %03.4f\n\n Target SBP : %03.4f, DBP : %03.4f\n Predicted SBP : %03.4f, DBP : %03.4f', arr(i,1), arr(i,2), arr(i,5), arr(i,6), arr(i,7), arr(i,8)));
    
%     set (h, 'Visible','off')s
    saveas(h, sprintf('%02dERROR%03d.png',error_range, i)); % png foramt
    saveas(h, sprintf('%02dERROR%03d.fig',error_range, i)); % will create FIG1, FIG2,...
    
    close (h)
end

end

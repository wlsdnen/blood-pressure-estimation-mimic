plot_range = 500;
error_range = 80;

for error_range = 30:5:80
    arr1 = eval(sprintf ('error%02ds', error_range));
    arr2 = eval(sprintf ('error%02dd', error_range));
    
    filename1 = sprintf ('error%02ds', error_range);
    filename2 = sprintf ('error%02dd', error_range);
    
    SaveError (filename1, Part_1, arr1, plot_range, error_range);
    SaveError (filename2, Part_1, arr2, plot_range, error_range);
end

function SaveError(filename, Part_1, arr, pic_range, error_range)
%SAVEERROR 이 함수의 요약 설명 위치
%   자세한 설명 위치

PPG_IDX = 1;
BP_IDX  = 2;
ECG_IDX = 3;

for i = 1:size(arr, 1)
    
    h=figure;
    
    set(h, 'Position', [100 100 1920 1080])
    
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
    
    suptitle(sprintf ('Error - SBP : %03.4f, DBP : %03.4f\nTarget - SBP : %03.4f, DBP : %03.4f\nPredicted - SBP : %03.4f, DBP : %03.4f', arr(i,1), arr(i,2), arr(i,5), arr(i,6), arr(i,7), arr(i,8)));
    
    saveas(h, sprintf('%s_%03d.png', filename, i)); % png foramt
    saveas(h, sprintf('%s_%03d.fig', filename, i)); % will create FIG1, FIG2,...
    
    close (h)
end

end

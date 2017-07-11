filename1 = 'mimic-part1-ecg.csv';
filename2 = 'mimic-part1-ppg.csv';
filename3 = 'mimic-part1-bp.csv';
filename4 = 'mimic-part1-bp-label.csv';
filename5 = 'mimic-part1-index.csv';
filename6 = 'mimic-part1-length.csv';

for i = 1:length(bp)
    
    dlmwrite (filename1, ecg{1,i}, '-append','precision','%.8f');
    dlmwrite (filename2, ppg{1,i}, '-append','precision','%.8f');
%     dlmwrite (filename3, bp{1,i}, '-append','precision','%.8f');

    if mod (i, 10000) == 0
        i
    end
    
end


dlmwrite (filename4, bp_label,'precision','%.8f');
dlmwrite (filename5, index);
dlmwrite (filename6, len);